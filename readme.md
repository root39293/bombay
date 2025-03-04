# Bombay

Bombay는 RAG(Retrieval-Augmented Generation) 파이프라인 구축용 Python 라이브러리. 다양한 임베딩 모델, 질의 모델, 벡터 데이터베이스 지원 및 문서 처리 기능 제공.

## 설치

```bash
pip install bombay
```

## 주요 기능

- **자동 모델 관리**: OpenAI API 모델 자동 관리 및 최신 모델 사용
- **다양한 벡터 데이터베이스**: HNSWLib, ChromaDB, Pinecone, pgvector 등 지원
- **문서 처리**: PDF, Word, Markdown, HTML 등 다양한 형식의 문서 처리
- **청크 분할 전략**: 단순, 단락, 문장, 의미 기반 등 다양한 청크 분할 전략 제공
- **플러그인 시스템**: 새로운 벡터 데이터베이스, 임베딩 모델, 문서 로더 등 쉬운 추가 가능

## 기본 사용법

### 파이프라인 생성

```python
from bombay.pipeline import create_pipeline
import os

# API 키 설정
api_key = os.getenv("OPENAI_API_KEY")

# 파이프라인 생성
pipeline = create_pipeline(
    embedding_model_name='openai',  # 자동으로 최신 임베딩 모델 사용
    query_model_name='gpt-3',       # 자동으로 최신 GPT 모델 사용
    vector_db='chromadb',           # ChromaDB 사용
    api_key=api_key,
    use_persistent_storage=True     # 데이터 영구 저장
)
```

### 문서 추가

```python
# 방법 1: 직접 문서 추가
documents = [
    "고양이는 포유류에 속하는 동물입니다.",
    "고양이는 약 6,000년 전부터 인간과 함께 살아온 것으로 추정됩니다."
]

# 메타데이터와 함께 문서 추가
metadatas = [
    {"source": "위키백과", "category": "동물"},
    {"source": "위키백과", "category": "역사"}
]

pipeline.add_documents(documents, metadatas)

# 방법 2: 파일에서 문서 추가
pipeline.process_file(
    "documents/article.pdf",
    chunking_strategy="paragraph",
    chunk_size=1000,
    chunk_overlap=200
)

# 방법 3: 디렉토리에서 문서 추가
pipeline.process_directory(
    "documents/",
    recursive=True,
    chunking_strategy="semantic"
)
```

### 질의 응답

```python
# 기본 검색
result = pipeline.search_and_answer(
    "고양이는 어떤 동물인가요?",
    k=2  # 상위 2개 문서 검색
)

print(f"질문: {result['query']}")
print(f"관련 문서: {result['relevant_docs']}")
print(f"답변: {result['answer']}")

# 필터링을 사용한 검색
result = pipeline.search_and_answer(
    "고양이의 역사에 대해 알려주세요.",
    k=1,
    filter_criteria={"category": "역사"}
)
```

### 프롬프트 템플릿 설정

```python
pipeline.set_prompt_template("""
다음 문서를 참고하여 질문에 정확하게 답변해주세요.

문서:
{relevant_docs}

질문: {query}

답변:
""")
```

### 청크 분할 전략 설정

```python
pipeline.set_chunking_strategy(
    strategy="sentence",  # 문장 기반 분할
    chunk_size=1000,
    chunk_overlap=200
)
```

## 벡터 데이터베이스 옵션

### HNSWLib (인메모리)

```python
pipeline = create_pipeline(
    embedding_model_name='openai',
    query_model_name='gpt-3',
    vector_db='hnswlib',
    api_key=api_key,
    similarity='cosine'
)
```

### ChromaDB (로컬 저장)

```python
pipeline = create_pipeline(
    embedding_model_name='openai',
    query_model_name='gpt-3',
    vector_db='chromadb',
    api_key=api_key,
    use_persistent_storage=True,
    collection_name='my_collection'
)
```

### Pinecone (클라우드)

```python
pipeline = create_pipeline(
    embedding_model_name='openai',
    query_model_name='gpt-3',
    vector_db='pineconedb',
    api_key=api_key,
    api_key_pinecone='YOUR_PINECONE_API_KEY',
    environment='us-west1-gcp',
    index_name='my-index',
    create_index=True
)
```

### pgvector (PostgreSQL)

```python
pipeline = create_pipeline(
    embedding_model_name='openai',
    query_model_name='gpt-3',
    vector_db='pgvectordb',
    api_key=api_key,
    connection_string='postgresql://username:password@localhost:5432/vectordb',
    table_name='embeddings',
    create_table=True
)
```

## 예제

더 많은 예제는 `example/` 디렉토리에서 확인 가능:

- `chromadb_example.py`: ChromaDB 사용 기본 예제
- `hnswlib_example.py`: HNSWLib 사용 기본 예제
- `document_processing_example.py`: 문서 처리 및 청크 분할 예제
- `vector_db_plugins_example.py`: 다양한 벡터 데이터베이스 플러그인 사용 예제

## 의존성

기본 의존성:
- openai>=1.0.0
- numpy>=1.20.0
- hnswlib>=0.7.0
- chromadb>=0.4.0
- python-dotenv>=1.0.0
- scikit-learn>=1.0.0
- beautifulsoup4>=4.10.0
- markdown>=3.4.0
- PyPDF2>=3.0.0
- python-docx>=0.8.11
- requests>=2.28.0

선택적 의존성:
- pinecone-client>=2.2.0 (Pinecone 사용 시)
- psycopg2-binary>=2.9.0 (pgvector 사용 시)
- pymupdf>=1.20.0 (PDF 이미지 추출 시)

## 라이선스

MIT

