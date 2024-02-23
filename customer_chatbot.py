#필요한 라이브러리#####################################################

import wave
import torch
import pickle 
import pyaudio
import IPython
import tiktoken
import numpy as np
from openai import OpenAI
from annoy import AnnoyIndex
from google.cloud import texttospeech
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain.chains import LLMChain
from langchain import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import warnings
warnings.filterwarnings('ignore')
################################################################

#.0. 필요 모델 호출
## SBERT Model
model_dir = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'
model_sbert = SentenceTransformer(model_dir)

## Reranker Model
tokenizer = AutoTokenizer.from_pretrained("Dongjin-kr/ko-reranker")
model = AutoModelForSequenceClassification.from_pretrained("Dongjin-kr/ko-reranker")
model.eval()

################################################################


def make_vector_db(metadata, model_sbert):
    """
    Annoy와 SBERT 모델을 통해 Vector DB를 만드는 함수
    metadata    : Q, A, keyword Dict
    model_sbert : BERT Model
    """
    # 1. Annoy 인덱스 초기화 (벡터 차원과 거리 측정 방법 설정)
    vector_dimension = 768  # BERT 벡터 차원
    annoy_quest = AnnoyIndex(vector_dimension, 'angular')
    annoy_key   = AnnoyIndex(vector_dimension, 'angular')

    # 2. Annoy Index 생성
    idx = 0
    for i in range(len(metadata)):
        question = metadata[i]['question']
        keyword  = metadata[i]['keyword']

        # 벡터 생성 및 Annoy 인덱스에 추가
        question_vector = model_sbert.encode(question)
        key_vector      = model_sbert.encode(keyword)
        annoy_quest.add_item(idx, question_vector)
        annoy_key.add_item(idx, key_vector)
        idx += 1

    # Annoy 인덱스 구축
    annoy_quest.build(10)  # 10개의 트리로 구축
    annoy_key.build(10)

    # DB 저장
    annoy_quest.save('./faq_vector_db.ann') # 경로 수정 필요
    annoy_key.save('./key_vector_db.ann')   # 경로 수정 필요
    print('DB Complete')


def recoding(WAVE_OUTPUT_FILENAME):
    """
    pyaudio 라이브러리를 이용하여 마이클를 통해 목소리 녹음
    WAVE_OUTPUT_FILENAME : Save Audio File Path
    """
    # 0. 녹음 설정
    FORMAT = pyaudio.paInt16  # 오디오 포맷
    CHANNELS = 1              # 모노 채널
    RATE = 44100              # 샘플링 레이트(44100Hz)
    CHUNK = 1024              # 프레임당 버퍼의 크기
    RECORD_SECONDS = 5        # 녹음할 시간(초)

    # 1. PyAudio 생성
    audio = pyaudio.PyAudio()

    # 2. 녹음 시작
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("녹음 중...")

    # 3. 지정된 시간 동안 오디오 데이터를 버퍼에 저장
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    # 4. 녹음 종료
    print("녹음 완료.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # 5. WAV 파일로 저장
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def speech_to_text_whisper(client, WAVE_OUTPUT_FILENAME):
    """
    OpenAI STT : Whisper 활용
    STT 모델을 활용하여 오디오 파일을 텍스트로 변환
    client               : OPEN AI Client
    WAVE_OUTPUT_FILENAME : Path Audio File
    """
    # 1. 오디오 파일을 텍스트로 변환
    audio_file= open(WAVE_OUTPUT_FILENAME, "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1",      # 모델명
        file=audio_file,        # 변환할 파일
        response_format="text"  # 변환 결과물 형태
    )
    print('변환된 텍스트 : ',transcript)
    return transcript


def create_chat(api_key):
    """
    GPT Client 생성 함수
    Clinet 기반 이전 K개까지 질문 같이 전송하는 ConversationChain 생성
    api_key : Open AI API Key
    """
    # 1. LLM 모델 생성
    llm = ChatOpenAI(model_name = 'gpt-3.5-turbo', openai_api_key=api_key, temperature=0) # temperature는 얼마나 신뢰도 있는 답변을 생성할지 설정(0은 최대한 보수적으로 답변)

    # 2. conversation chain 생성
    memory = ConversationBufferWindowMemory(llm=llm, k=2) # k는 이전 대화를 몇개까지 기억할지 설정값
    conversation = ConversationChain(llm=llm,
                                     memory=memory,
                                     verbose=False)
    return conversation



def chat_generate(query, conversation,
                  model_sbert, annoy_index, annoy_key, metadata):
    """
    고객 질문에 대한 LLM 답변 생성
    query        : New Question
    conversation : ConversationChain of LangChain(GPT)
    model_sbert  : BERT Model
    annoy_index  : Annoy of Question DB
    annoy_key    : Annoy of Keyword DB
    metadata     : Dict of Q, A, Keyword
    """
    # 0. 대화 종료 구문
    if query == '':
        print('대화가 종료됩니다.')
    else:
        # 1. 유사 질문 탐색(RAG)
        score, answer, keyword, question = find_answer_text(query, model_sbert,
                                                       annoy_index, annoy_key, metadata)
    # 2. Generated Knowledge Prompting
    my_template = (
        "#당신은 크레딧커넥트의 고객 상담원입니다. 높임말을 사용하여 2~3문장으로 답변하세요.#\n"
        "Question : {x}\n"
        "Knowledge : {answer}\n"
        "Answer : "
    )
    prompt = PromptTemplate.from_template(my_template)

    # 3. 답변 생성
    result = conversation.predict(input = prompt.format(x=query,answer=answer))
    print('답변 : ',result)
    return result


def get_top_n(list_a, num):
    '''
    리스트 변수에서 상위 num개를 추출하는 코드
    list_a : [int, int, ...] 
    num    : int, 추출하고 싶은 개수 
    '''
    tmp = list_a.copy()
    tmp.sort()
   
    top_num = tmp[-num:]    # 뒤에서부터 추출
    top_idx = [list_a.index(x) for x in top_num]
    return top_num, top_idx

def exp_normalize(x):
    """
    유사도 점수 변환하는 코드
    x : score
    """
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

def find_answer_text(new_question, annoy_index, annoy_key, metadata, k=20):
    """
    new_question : New Question
    annoy_index  : Annoy of Questions
    annoy_key    : Annoy of Keywords
    metadata     : Dict of Q, A, Keyword
    k            : 벡터 DB에서 추출할 후보 개수
    """
    # 1) 신규변수 Embedding
    new_question_vector = model_sbert.encode(new_question, batch_size=512)

    # 2) Vector 기반 유사 질문 추출
    question_result = annoy_index.get_nns_by_vector(new_question_vector, k, include_distances=True)
    key_result = annoy_key.get_nns_by_vector(new_question_vector, k, include_distances=True)

    # 3) 키워드 유사도와 질문 유사도 교차 검증
    and_idx   = list(set(question_result[0])&set(key_result[0]))

    # 4) 최종 유사도 후보 변환(신규 질문 : 유사 질문)
    quest_search = [[new_question, metadata[i]['question']] for i in and_idx]
    key_search = [[new_question, metadata[i]['keyword']] for i in and_idx]
    # 5) Reranker 유사도 산출
    input_q = tokenizer(quest_search, padding=True, truncation=True, return_tensors='pt', max_length=512)
    input_k = tokenizer(key_search, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores_q = model(**input_q, return_dict=True).logits.view(-1, ).float()
    scores_k = model(**input_k, return_dict=True).logits.view(-1, ).float()
    # 6) 점수 변환
    scores_q = exp_normalize(scores_q.detach().numpy())
    scores_k = exp_normalize(scores_k.detach().numpy())
    # 7) 점수 통합 및 상위 질문 추출
    scores = scores_q + scores_k
    top_num_q, top_idx_q = get_top_n(list(scores), 3) # 유사 질문 3개 추출
    result_idx = [and_idx[i] for i in top_idx_q]
    return top_num_q, result_idx


def text_to_speech_google(client, gen_result, AUDIO_OUTPUT):
    """
    Text to Speech DEF
    client       : Google Cloud Client
    gen_result   : LLM Result 
    AUDIO_OUTPUT : Save File Path
    """
    # 1. Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=gen_result)

    # 2. Build the voice request, select the language code ("en-US") and the ssml
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",
        name="ko-KR-Standard-B",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    )

    # 3. Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # 4. Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # 5. The response's audio_content is binary.
    with open(AUDIO_OUTPUT, "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
