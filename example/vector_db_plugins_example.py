# vector_db_plugins_example.py
from bombay.pipeline import create_pipeline
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Pinecone 설정 (환경 변수에서 로드)
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "bombay-example")

# pgvector 설정 (환경 변수에서 로드)
pg_connection_string = os.getenv("PG_CONNECTION_STRING", 
                                "postgresql://username:password@localhost:5432/vectordb")

print("벡터 데이터베이스 플러그인 예제")
print("-" * 50)

# 예제 문서
documents = [
    "벡터 데이터베이스는 벡터 임베딩을 효율적으로 저장하고 검색하는 데이터베이스입니다.",
    "Pinecone은 클라우드 기반의 벡터 데이터베이스로, 확장성이 뛰어납니다.",
    "pgvector는 PostgreSQL의 확장으로, 관계형 데이터베이스에 벡터 검색 기능을 추가합니다.",
    "Hnswlib은 인메모리 벡터 데이터베이스로, 빠른 검색 속도를 제공합니다.",
    "ChromaDB는 로컬 및 클라우드 환경에서 모두 사용할 수 있는 벡터 데이터베이스입니다."
]

# 1. Pinecone 예제 (API 키가 있는 경우에만 실행)
if pinecone_api_key:
    try:
        print("\n1. Pinecone 벡터 데이터베이스 예제")
        print("-" * 40)
        
        # Pinecone 파이프라인 생성
        pinecone_pipeline = create_pipeline(
            embedding_model_name='openai',
            query_model_name='gpt-4',
            vector_db='pineconedb',  # 플러그인 이름
            api_key=api_key,
            # Pinecone 특정 매개변수
            api_key_pinecone=pinecone_api_key,
            environment=pinecone_environment,
            index_name=pinecone_index_name,
            create_index=True  # 인덱스가 없는 경우 생성
        )
        
        # 문서 추가
        pinecone_pipeline.add_documents(documents)
        
        # 메타데이터와 함께 문서 추가
        metadatas = [{"source": "article", "topic": "vector_db"} for _ in documents]
        pinecone_pipeline.add_documents(documents, metadatas)
        
        # 질의 응답
        query = "Pinecone의 특징은 무엇인가요?"
        result = pinecone_pipeline.search_and_answer(query, k=2)
        
        print(f"질문: {result['query']}")
        print(f"관련 문서: {result['relevant_docs']}")
        print(f"답변: {result['answer']}")
        
        # 필터링을 사용한 검색
        query = "벡터 데이터베이스의 종류는 무엇인가요?"
        result = pinecone_pipeline.search_and_answer(
            query, 
            k=3,
            filter_criteria={"source": "article"}
        )
        
        print(f"\n필터링 검색 결과:")
        print(f"질문: {result['query']}")
        print(f"관련 문서: {result['relevant_docs']}")
        print(f"답변: {result['answer']}")
        
    except Exception as e:
        print(f"Pinecone 예제 실행 중 오류 발생: {e}")
        print("Pinecone을 사용하려면 'pip install pinecone-client'를 실행하여 패키지를 설치하세요.")
else:
    print("\nPinecone API 키가 설정되지 않았습니다. PINECONE_API_KEY 환경 변수를 설정하세요.")

# 2. pgvector 예제 (연결 문자열이 있는 경우에만 실행)
if pg_connection_string and "username:password" not in pg_connection_string:
    try:
        print("\n2. pgvector 벡터 데이터베이스 예제")
        print("-" * 40)
        
        # pgvector 파이프라인 생성
        pgvector_pipeline = create_pipeline(
            embedding_model_name='openai',
            query_model_name='gpt-4',
            vector_db='pgvectordb',  # 플러그인 이름
            api_key=api_key,
            # pgvector 특정 매개변수
            connection_string=pg_connection_string,
            table_name="bombay_examples",
            create_table=True,  # 테이블이 없는 경우 생성
            similarity="cosine"
        )
        
        # 문서 추가
        pgvector_pipeline.add_documents(documents)
        
        # 질의 응답
        query = "pgvector의 특징은 무엇인가요?"
        result = pgvector_pipeline.search_and_answer(query, k=2)
        
        print(f"질문: {result['query']}")
        print(f"관련 문서: {result['relevant_docs']}")
        print(f"답변: {result['answer']}")
        
    except Exception as e:
        print(f"pgvector 예제 실행 중 오류 발생: {e}")
        print("pgvector를 사용하려면 'pip install psycopg2-binary'를 실행하여 패키지를 설치하세요.")
else:
    print("\npgvector 연결 문자열이 올바르게 설정되지 않았습니다. PG_CONNECTION_STRING 환경 변수를 설정하세요.")

# 3. 기본 벡터 데이터베이스 비교
print("\n3. 기본 벡터 데이터베이스 비교")
print("-" * 40)

# HNSWLib 파이프라인 생성
hnswlib_pipeline = create_pipeline(
    embedding_model_name='openai',
    query_model_name='gpt-4',
    vector_db='hnswlib',
    api_key=api_key
)

# ChromaDB 파이프라인 생성
chromadb_pipeline = create_pipeline(
    embedding_model_name='openai',
    query_model_name='gpt-4',
    vector_db='chromadb',
    api_key=api_key,
    use_persistent_storage=True
)

# 문서 추가
hnswlib_pipeline.add_documents(documents)
chromadb_pipeline.add_documents(documents)

# 질의 응답 및 성능 비교
query = "벡터 데이터베이스의 종류와 특징을 설명해주세요."

print("\nHNSWLib 결과:")
result = hnswlib_pipeline.search_and_answer(query, k=3)
print(f"관련 문서 수: {len(result['relevant_docs'])}")
print(f"답변: {result['answer']}")

print("\nChromaDB 결과:")
result = chromadb_pipeline.search_and_answer(query, k=3)
print(f"관련 문서 수: {len(result['relevant_docs'])}")
print(f"답변: {result['answer']}") 