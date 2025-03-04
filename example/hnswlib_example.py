# hnswlib_example.py
from bombay.pipeline import create_pipeline
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 파이프라인 생성 (HNSWLib 사용)
pipeline = create_pipeline(
    embedding_model_name='openai',
    query_model_name='gpt-3',
    vector_db='hnswlib',
    api_key=api_key,
    similarity='cosine'  # 코사인 유사도 사용
)

# 청크 분할 전략 설정
pipeline.set_chunking_strategy(
    strategy="sentence",  # 문장 기반 분할
    chunk_size=1000,
    chunk_overlap=200
)

# 디렉토리에서 문서 처리 (디렉토리가 존재한다고 가정)
# pipeline.process_directory(
#     "example/documents",
#     recursive=True,
#     chunking_strategy="semantic"  # 의미 기반 분할
# )

# 직접 문서 추가
documents = [
    "인공지능(AI)은 인간의 학습, 추론, 지각, 문제 해결 능력 등을 컴퓨터 프로그램으로 구현한 기술입니다.",
    "머신러닝은 컴퓨터가 데이터로부터 학습하여 패턴을 인식하고 예측하는 AI의 한 분야입니다.",
    "딥러닝은 인공 신경망을 기반으로 하는 머신러닝의 한 종류로, 복잡한 패턴을 인식하는 데 뛰어납니다.",
    "자연어 처리(NLP)는 컴퓨터가 인간의 언어를 이해하고 처리하는 AI 기술입니다.",
    "컴퓨터 비전은 컴퓨터가 이미지나 비디오를 이해하고 해석하는 AI 기술입니다."
]

pipeline.add_documents(documents)

# 질의 응답
queries = [
    "인공지능이란 무엇인가요?",
    "머신러닝과 딥러닝의 차이점은 무엇인가요?",
    "자연어 처리란 무엇인가요?"
]

for i, query in enumerate(queries):
    result = pipeline.search_and_answer(query, k=2)
    
    print(f"\n질문 {i+1}:")
    print(f"질문: {result['query']}")
    print(f"관련 문서:")
    for j, doc in enumerate(result['relevant_docs']):
        print(f"  {j+1}. {doc} (유사도: {result['distances'][j]:.4f})")
    print(f"답변: {result['answer']}")

# 임베딩 모델 정보 출력
print(f"\n현재 사용 중인 임베딩 모델: {pipeline.embedding_model.model}")
print(f"임베딩 차원: {pipeline.embedding_model.get_dimension()}")

# 질의 모델 정보 출력
print(f"\n현재 사용 중인 질의 모델: {pipeline.query_model.model}")
print(f"생성 온도: {pipeline.query_model.temperature}")