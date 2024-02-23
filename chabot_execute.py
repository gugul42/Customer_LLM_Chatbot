####################################################################
import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
import urllib.request
from tqdm import tqdm
from openai import OpenAI
from google.cloud import texttospeech
from customer_chatbot import *

import tiktoken
from langchain import OpenAI, ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

import warnings
warnings.filterwarnings('ignore')
####################################################################


# DB Load
vector_dimension = 768  # 벡터 차원
annoy_index = AnnoyIndex(vector_dimension, 'angular')
annoy_key   = AnnoyIndex(vector_dimension, 'angular')
annoy_index.load('./vector_db/faq_vector_db.ann')  # 저장된 인덱스 파일 로드
annoy_key.load('./vector_db/key_vector_db.ann')  # 저장된 인덱스 파일 로드

#탬플릿 데이터 로드
template = pd.read_excel('./data/chatbot_template.xlsx')

# 템플릿 Dict 형태로 변환
metadata = []
for _, row in template.iterrows():
    # 질문, 답변, 검색베이스(키워드)로 나눠 저장
    temp = {'question':row['질문 형태'],
            'answer':row['답변'] ,
            'keyword':row['검색베이스']
            }
    metadata.append(temp)

# LLM Load
api_key = '' # Open API 
client = OpenAI(api_key = api_key)
llm = ChatOpenAI(model_name = 'gpt-3.5-turbo', openai_api_key=api_key, temperature=0)


# the conversation chain
memory = ConversationBufferWindowMemory(llm=llm, k=2)
conversation = ConversationChain(llm=llm,
                                 memory=memory,
                                 verbose=False)

def execute_chatbot(query):
    # 대화 종료 구문
    if query == '':
        result = '대화가 종료됩니다.'
        print('대화가 종료됩니다.')
    else:
        # 유사 질문 탐색
        top_num, result_idx = find_answer_text(query, annoy_index, annoy_key, metadata)

        # Generated Knowledge Prompting : 6번
        my_template = (
            "#당신은 크레딧커넥트의 고객 상담원입니다. 높임말을 사용하여 답변하세요.#\n"
            "Question : {x1}\n"
            "Knowledge : {answer1}\n"
            "Question : {x2}\n"
            "Knowledge : {answer2}\n"
            "Question : {x3}\n"
            "Knowledge : {answer3}\n"
            "Question : {query}\n"
            "Answer : "
        )
        prompt = PromptTemplate.from_template(my_template)

        # 유사한 질문이 없는 경우 
        if min(top_num) > 0.3:
            result = '고객님 죄송합니다. 해당 문제를 보다 잘 해결하기 위해서는 상세한 설명이 필요합니다.\n자세한 설명을 작성해주시거나 혹은 고객센터(02-761-0171 또는 02-761-0172)로 연락주세요.\n'
            print(result)
        else:
            # 답변 생성 
            prompt_result = prompt.format(query=query,
                            x1=metadata[result_idx[0]]['question'],answer1=metadata[result_idx[0]]['answer'],
                            x2=metadata[result_idx[1]]['question'],answer2=metadata[result_idx[1]]['answer'],
                            x3=metadata[result_idx[2]]['question'],answer3=metadata[result_idx[2]]['answer'])

            result = conversation.predict(input = prompt_result)
            print('질문 : ',query)
            print('답변 : ',result)
            print('\n')
    return result
            

def execute_callbot(client):
    ## 1. Recoding
    WAVE_OUTPUT_FILENAME = "./audio/recoding.wav" # 녹음 저장할 위치
    recoding(WAVE_OUTPUT_FILENAME)

    # 2. Speech to Text
    recoding_data = speech_to_text_whisper(client, WAVE_OUTPUT_FILENAME)
    
    if ('금리' in recoding_data)|('한도' in recoding_data)|('대출' in recoding_data):
        gen_result = execute_loanbot(query)
    else:
        # 3. LLM
        gen_result = execute_chatbot(recoding_data)

    # 4. Text to Speech
    AUDIO_OUTPUT = "./audio/result.mp3"
    api_path = "../ccm-customer-46e2457b7dfa.json"# Google TTS KEY
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=api_path
    client = texttospeech.TextToSpeechClient()
    text_to_speech_google(client, gen_result, AUDIO_OUTPUT)
    IPython.display.Audio(AUDIO_OUTPUT)
        
        
def execute_loanbot(query):
    # 대화 종료 구문
    if query == '':
        result = '대화가 종료됩니다.'
        print('대화가 종료됩니다.')
    else:
        interest = pd.read_csv('./data/interest.csv')
        limit = pd.read_csv('./data/limit_by.csv')
        customer_info = pd.read_csv('./data/seller_infomation.csv')

        # 데이터 변환
        customer_info = customer_info.sum().reset_index()
        customer_info['ratio'] = customer_info[0]/customer_info[0].sum()

        # 뱅크별 금리 한도 조합
        banks = ['키움','도이치','KB']
        banks_dict = {}
        for b in banks:
            b_limit = 0
            b_inter = 0
            for i in range(len(limit[b])):
                b_limit += float(limit[b][i][:-1])/100*customer_info[0][i]
                b_inter += float(interest[b][i][:-1])/100*customer_info['ratio'][i]
            banks_dict[b]= [b_limit,b_inter]


        # Generated Knowledge Prompting
        my_template = (
            "#당신은 크레딧커넥트의 고객 상담원입니다. 높임말을 사용하여 답변하세요.#\n"
            "다음은 대출 정보입니다.\n"
            "{bank1} 은행의 한도는 {limit1}만원, {bank2} 은행의 한도는 =  {limit2}만원, {bank3} 은행의 한도는 = {limit3}만원\n"
            "{bank1} 은행의 금리는 {inter1:.2f}%, {bank2} 은행의 금리는 =  {inter2:.2f}%, {bank3} 은행의 금리는 = {inter3:.2f}%\n"
            "고객의 질문입니다 : {query}"
            "Answer : "
        )
        prompt = PromptTemplate.from_template(my_template)

        # 답변 생성 
        banks = list(banks_dict.keys())
        prompt_result = prompt.format(query=query,
                        bank1=banks[0],limit1=banks_dict[banks[0]][0],inter1=banks_dict[banks[0]][1]*100,
                        bank2=banks[1],limit2=banks_dict[banks[1]][0],inter2=banks_dict[banks[1]][1]*100,
                        bank3=banks[2],limit3=banks_dict[banks[2]][0],inter3=banks_dict[banks[2]][1]*100)

        result = conversation.predict(input = prompt_result)
        print('질문 : ',query)
        print('답변 : ',result)
        print('\n')
    return result
