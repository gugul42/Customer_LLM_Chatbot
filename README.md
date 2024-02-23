# 고객센터 챗봇(Rag+Reranker+GPT3.5) <br>

![Untitled](https://github.com/gugul42/customer_llm_chatbot/assets/98931388/fe157808-4f10-44c1-a4f9-f880c9808638)

### 1) Summary
- GPT 3.5 Turbo를 사용하는 챗봇 및 콜봇 모델
    - GPT, Llama, Yi 등 기능 테스트 진행 결과 GPT의 성능이 가장 좋았음
- RAG 기능을 위해 FAQ와 민원 상담 데이터 Vector DB화(보안상 문제로 업로드X)
- Annoy, Faiss 기능 테스트 진행 결과, 작은 데이터베이스 상에서는 큰 문제가 없어 Annoy 사용
- SBERT + Reranker Model을 결합하여 가장 유사한 질문 및 문서 찾을 수 있음
- LangChain을 활용한 Vector DB+LLM(GPT 3.5) 챗봇 모델 개발
- Whisper STT 활용한 콜봇 음성인식 기능 구현
- GoogleCloud TTS 활용한 콜봇 구현
<br/>


### 2) Process(고객센터)
<b> 1. 사용자 접속</b>
- 사용자가 문의를 위해 고객센터 이용
  
<b> 2. 고객센터 전화</b>
- 상담원이 모두 상담 중이거나, 부재 중인경우 Call 대기봇 연결
- Call 대기봇에게 문의사항 전달(10초 내외)
- 문의사항 STT 모델 기반으로 텍스트 변환
- 변환된 텍스트와 전화번호 상담원에게 전달
- 상담원 내용 파악 후 아웃바운드
  
<b> 3. 고객센터 챗봇 이용</b>
- 문의사항 챗봇 전달
- Embedding Model(SBERT)을 통해 Vector 변환
- Annoy Vector DB를 이용하여 유사한 질문 및 상담내용 20개 후보 추출
- Reranker Model을 통해 최종 유사한 질문과 답변 3개 추출
- 유사 질문 3개 기반 프롬프트 엔지니어링(LangChain)
- GPT 3.5 Turbo 모델에 API 기반 전송 및 답변 산출(OpenAI)
- 고객에게 답변 전송
