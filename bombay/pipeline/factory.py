import logging
from typing import Dict, List, Optional, Union, Any

from bombay.models.embedding_model import EmbeddingModel, OpenAIEmbedding
from bombay.models.query_model import QueryModel, OpenAIQuery
from bombay.vector_db.vector_db import VectorDB
from bombay.vector_db.hnswlib_db import HNSWLib
from bombay.vector_db.chroma_db import ChromaDB
from bombay.pipeline.rag_pipeline import RAGPipeline
from bombay.document_processing import DocumentProcessor

logger = logging.getLogger(__name__)

def create_pipeline(
    embedding_model_name: str = "openai",
    query_model_name: str = "gpt-3",
    vector_db: str = "hnswlib",
    api_key: Optional[str] = None,
    **kwargs
) -> RAGPipeline:
    """
    RAG 파이프라인 생성
    
    Args:
        embedding_model_name: 임베딩 모델 이름 (기본값: "openai")
        query_model_name: 질의 모델 이름 (기본값: "gpt-3")
        vector_db: 벡터 데이터베이스 유형 (기본값: "hnswlib")
        api_key: OpenAI API 키
        **kwargs: 추가 매개변수
        
    Returns:
        RAGPipeline 인스턴스
    """
    # 임베딩 모델 생성
    embedding_model = create_embedding_model(embedding_model_name, api_key, **kwargs)
    
    # 질의 모델 생성
    query_model = create_query_model(query_model_name, api_key, **kwargs)
    
    # 벡터 데이터베이스 생성
    vector_database = create_vector_db(vector_db, embedding_model.get_dimension(), **kwargs)
    
    # 문서 처리기 생성
    document_processor = DocumentProcessor(embedding_model=embedding_model)
    
    # RAG 파이프라인 생성
    pipeline = RAGPipeline(
        embedding_model=embedding_model,
        query_model=query_model,
        vector_db=vector_database,
        document_processor=document_processor
    )
    
    return pipeline


def create_embedding_model(
    model_name: str = "openai",
    api_key: Optional[str] = None,
    **kwargs
) -> EmbeddingModel:
    """
    임베딩 모델 생성
    
    Args:
        model_name: 모델 이름 (기본값: "openai")
        api_key: API 키
        **kwargs: 추가 매개변수
        
    Returns:
        EmbeddingModel 인스턴스
    """
    if model_name.lower() in ["openai", "auto", "text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"]:
        # OpenAI 임베딩 모델 생성
        return OpenAIEmbedding(model=model_name, api_key=api_key, **kwargs)
    else:
        # 기본값: OpenAI 임베딩 모델
        logger.warning(f"알 수 없는 임베딩 모델: {model_name}, OpenAI 모델을 사용합니다.")
        return OpenAIEmbedding(api_key=api_key, **kwargs)


def create_query_model(
    model_name: str = "gpt-3",
    api_key: Optional[str] = None,
    **kwargs
) -> QueryModel:
    """
    질의 모델 생성
    
    Args:
        model_name: 모델 이름 (기본값: "gpt-3")
        api_key: API 키
        **kwargs: 추가 매개변수
        
    Returns:
        QueryModel 인스턴스
    """
    # 온도 설정
    temperature = kwargs.pop("temperature", 0.7)
    
    # 최대 토큰 수 설정
    max_tokens = kwargs.pop("max_tokens", None)
    
    if model_name.lower() in ["auto", "gpt-3", "gpt3", "gpt-4", "gpt4", "gpt-4o", "gpt4o", "gpt-4o-mini", 
                             "gpt-4.5-preview", "o1", "o1-mini", "o3-mini", "reasoning"]:
        # OpenAI 질의 모델 생성
        return OpenAIQuery(model=model_name, api_key=api_key, 
                          temperature=temperature, max_tokens=max_tokens, **kwargs)
    elif model_name.startswith("gpt-") or model_name.startswith("o"):
        # OpenAI 질의 모델 생성 (구체적인 모델 이름)
        return OpenAIQuery(model=model_name, api_key=api_key, 
                          temperature=temperature, max_tokens=max_tokens, **kwargs)
    else:
        # 기본값: OpenAI 질의 모델
        logger.warning(f"알 수 없는 질의 모델: {model_name}, GPT-3 모델을 사용합니다.")
        return OpenAIQuery(model="gpt-3", api_key=api_key, 
                          temperature=temperature, max_tokens=max_tokens, **kwargs)


def create_vector_db(
    db_type: str = "hnswlib",
    dimension: int = 1536,
    **kwargs
) -> VectorDB:
    """
    벡터 데이터베이스 생성
    
    Args:
        db_type: 데이터베이스 유형 (기본값: "hnswlib")
        dimension: 벡터 차원
        **kwargs: 추가 매개변수
        
    Returns:
        VectorDB 인스턴스
    """
    db_type = db_type.lower()
    
    if db_type == "hnswlib":
        # HNSWLib 벡터 데이터베이스 생성
        return HNSWLib(dim=dimension, **kwargs)
    
    elif db_type == "chromadb":
        # ChromaDB 벡터 데이터베이스 생성
        return ChromaDB(dim=dimension, **kwargs)
    
    elif db_type in ["pineconedb", "pinecone"]:
        # Pinecone 벡터 데이터베이스 생성
        try:
            from bombay.plugins.vector_db.pinecone_db import PineconeDB
            return PineconeDB(dim=dimension, **kwargs)
        except ImportError:
            logger.error("Pinecone 벡터 데이터베이스를 사용하려면 'pip install pinecone-client'를 실행하여 패키지를 설치하세요.")
            raise ImportError("Pinecone 벡터 데이터베이스를 사용하려면 'pip install pinecone-client'를 실행하여 패키지를 설치하세요.")
    
    elif db_type in ["pgvectordb", "pgvector"]:
        # pgvector 벡터 데이터베이스 생성
        try:
            from bombay.plugins.vector_db.pgvector_db import PGVectorDB
            return PGVectorDB(dim=dimension, **kwargs)
        except ImportError:
            logger.error("pgvector 벡터 데이터베이스를 사용하려면 'pip install psycopg2-binary'를 실행하여 패키지를 설치하세요.")
            raise ImportError("pgvector 벡터 데이터베이스를 사용하려면 'pip install psycopg2-binary'를 실행하여 패키지를 설치하세요.")
    
    else:
        # 기본값: HNSWLib 벡터 데이터베이스
        logger.warning(f"알 수 없는 벡터 데이터베이스 유형: {db_type}, HNSWLib를 사용합니다.")
        return HNSWLib(dim=dimension, **kwargs)


def run_pipeline(
    query: str,
    pipeline: RAGPipeline,
    k: int = 3,
    filter_criteria: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    파이프라인 실행
    
    Args:
        query: 질의 텍스트
        pipeline: RAG 파이프라인
        k: 검색할 문서 수
        filter_criteria: 필터링 기준
        **kwargs: 추가 매개변수
        
    Returns:
        결과 딕셔너리
    """
    return pipeline.search_and_answer(query, k=k, filter_criteria=filter_criteria, **kwargs)


def process_documents(
    documents: List[str],
    pipeline: RAGPipeline,
    metadatas: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> None:
    """
    문서 처리 및 파이프라인에 추가
    
    Args:
        documents: 문서 목록
        pipeline: RAG 파이프라인
        metadatas: 메타데이터 목록
        **kwargs: 추가 매개변수
    """
    pipeline.add_documents(documents, metadatas, **kwargs) 