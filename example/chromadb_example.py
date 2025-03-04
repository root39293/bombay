#chromadb_example.py
from bombay.pipeline import create_pipeline
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 파이프라인 생성 (ChromaDB 사용)
pipeline = create_pipeline(
    embedding_model_name='openai',  # 자동으로 최신 임베딩 모델 사용
    query_model_name='gpt-3',       # 자동으로 최신 GPT 모델 사용
    vector_db='chromadb',
    api_key=api_key,
    use_persistent_storage=True,    # 데이터 영구 저장
    collection_name='example_collection'
)

# 사용자 정의 프롬프트 템플릿 설정
pipeline.set_prompt_template("""
다음 문서를 참고하여 질문에 정확하게 답변해주세요.

문서:
{relevant_docs}

질문: {query}

답변:
""")

# 문서 추가 방법 1: 직접 문서 추가
documents = [
    "고양이는 포유류에 속하는 동물입니다.",
    "고양이는 약 6,000년 전부터 인간과 함께 살아온 것으로 추정됩니다.",
    "고양이는 예민한 청각과 후각을 가지고 있어 작은 움직임이나 냄새도 쉽게 감지할 수 있습니다."
]

# 메타데이터와 함께 문서 추가
metadatas = [
    {"source": "위키백과", "category": "동물", "topic": "고양이 특성"},
    {"source": "위키백과", "category": "역사", "topic": "고양이 역사"},
    {"source": "동물백과", "category": "동물", "topic": "고양이 특성"}
]

pipeline.add_documents(documents, metadatas)

# 문서 추가 방법 2: 파일에서 문서 추가 (파일이 존재한다고 가정)
# pipeline.process_file(
#     "example/cat_facts.pdf",
#     chunking_strategy="paragraph",
#     chunk_size=500,
#     chunk_overlap=100
# )

# 질의 응답 (기본)
query = "고양이는 어떤 동물인가요?"
result = pipeline.search_and_answer(query, k=2)

print("\n기본 검색 결과:")
print(f"질문: {result['query']}")
print(f"관련 문서: {result['relevant_docs']}")
print(f"답변: {result['answer']}")

# 필터링을 사용한 질의 응답
query = "고양이의 역사에 대해 알려주세요."
result = pipeline.search_and_answer(
    query, 
    k=1,
    filter_criteria={"category": "역사"}
)

print("\n필터링 검색 결과:")
print(f"질문: {result['query']}")
print(f"관련 문서: {result['relevant_docs']}")
print(f"답변: {result['answer']}")

# 사용 가능한 임베딩 모델 확인
available_models = pipeline.embedding_model.get_available_models()
print(f"\n사용 가능한 임베딩 모델: {available_models}")

# 사용 가능한 질의 모델 확인
available_models = pipeline.query_model.get_available_models()
print(f"\n사용 가능한 질의 모델: {available_models}")