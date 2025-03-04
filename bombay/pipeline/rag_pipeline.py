# bombay/pipeline/rag_pipeline.py

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import os
from .vector_db import VectorDB, HNSWLib, ChromaDB
from .embedding_models import EmbeddingModel, OpenAIEmbedding
from .query_models import QueryModel, OpenAIQuery
from ..utils.logging import logger
from ..utils.preprocessing import preprocess_text
from ..document_processing import DocumentProcessor, Chunk, Document

# RAG 파이프라인 클래스
class RAGPipeline:
    """RAG(Retrieval-Augmented Generation) 파이프라인 클래스"""
    
    def __init__(self, 
                 embedding_model: EmbeddingModel,
                 query_model: QueryModel,
                 vector_db: VectorDB,
                 document_processor: Optional[DocumentProcessor] = None,
                 **kwargs):
        """
        RAG 파이프라인 초기화
        
        Args:
            embedding_model: 임베딩 모델
            query_model: 질의 모델
            vector_db: 벡터 데이터베이스
            document_processor: 문서 처리기 (기본값: None)
            **kwargs: 추가 매개변수
        """
        self.embedding_model = embedding_model
        self.query_model = query_model
        self.vector_db = vector_db
        self.document_processor = document_processor or DocumentProcessor(embedding_model=embedding_model)
        self.kwargs = kwargs
        self.prompt_template = None
        
    def add_documents(self, 
                      documents: List[str], 
                      metadatas: Optional[List[Dict[str, Any]]] = None,
                      chunk_size: int = 1000,
                      chunk_overlap: int = 200,
                      **kwargs) -> List[str]:
        """
        문서 추가
        
        Args:
            documents: 문서 목록
            metadatas: 메타데이터 목록
            chunk_size: 청크 크기
            chunk_overlap: 청크 중첩 크기
            **kwargs: 추가 매개변수
            
        Returns:
            추가된 문서의 ID 목록
        """
        if not documents:
            return []
        
        # 메타데이터 확인
        if metadatas is None:
            metadatas = [{} for _ in range(len(documents))]
        elif len(metadatas) != len(documents):
            raise ValueError(f"문서 수({len(documents)})와 메타데이터 수({len(metadatas)})가 일치하지 않습니다.")
        
        # 문서 처리
        processed_documents = []
        processed_metadatas = []
        
        for i, doc in enumerate(documents):
            # 문서 객체 생성
            document = Document(doc, metadatas[i])
            
            # 문서 청킹
            chunks = self.document_processor.process_document(document, chunk_size, chunk_overlap)
            
            # 청크 추가
            for chunk in chunks:
                processed_documents.append(chunk.content)
                processed_metadatas.append(chunk.metadata)
        
        # 임베딩 생성
        vectors = self.embedding_model.get_embeddings(processed_documents)
        
        # 벡터 데이터베이스에 추가
        ids = self.vector_db.add(
            vectors=vectors,
            documents=processed_documents,
            metadatas=processed_metadatas,
            **kwargs
        )
        
        return ids
    
    def search(self, 
               query: str, 
               k: int = 4, 
               filter_criteria: Optional[Dict[str, Any]] = None,
               **kwargs) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """
        검색 수행
        
        Args:
            query: 검색 쿼리
            k: 검색할 문서 수
            filter_criteria: 필터링 기준
            **kwargs: 추가 매개변수
            
        Returns:
            (문서 목록, 거리 목록, 메타데이터 목록) 튜플
        """
        # 쿼리 임베딩 생성
        query_vector = self.embedding_model.get_embedding(query)
        
        # 벡터 데이터베이스 검색
        documents, distances, metadatas = self.vector_db.search(
            query_vector=query_vector,
            k=k,
            filter_criteria=filter_criteria,
            **kwargs
        )
        
        return documents, distances, metadatas
    
    def generate(self, 
                 query: str, 
                 context: List[str],
                 **kwargs) -> str:
        """
        응답 생성
        
        Args:
            query: 질의 텍스트
            context: 컨텍스트 문서 목록
            **kwargs: 추가 매개변수
            
        Returns:
            생성된 응답
        """
        # 질의 모델로 응답 생성
        response = self.query_model.generate(
            query=query,
            context=context,
            **kwargs
        )
        
        return response
    
    def search_and_answer(self, 
                          query: str, 
                          k: int = 4, 
                          filter_criteria: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        검색 및 응답 생성
        
        Args:
            query: 질의 텍스트
            k: 검색할 문서 수
            filter_criteria: 필터링 기준
            **kwargs: 추가 매개변수
            
        Returns:
            결과 딕셔너리
        """
        # 검색 수행
        documents, distances, metadatas = self.search(
            query=query,
            k=k,
            filter_criteria=filter_criteria
        )
        
        if not documents:
            return {
                "query": query,
                "answer": "검색 결과가 없습니다.",
                "documents": [],
                "distances": [],
                "metadatas": []
            }
        
        # 응답 생성
        answer = self.generate(
            query=query,
            context=documents,
            **kwargs
        )
        
        # 결과 반환
        return {
            "query": query,
            "answer": answer,
            "documents": documents,
            "distances": distances,
            "metadatas": metadatas
        }
    
    def save(self, path: str) -> None:
        """
        파이프라인 저장
        
        Args:
            path: 저장 경로
        """
        # 벡터 데이터베이스 저장
        self.vector_db.save(path)
    
    def load(self, path: str) -> None:
        """
        파이프라인 로드
        
        Args:
            path: 로드 경로
        """
        # 벡터 데이터베이스 로드
        self.vector_db.load(path)

    def set_prompt_template(self, template: str) -> None:
        """
        프롬프트 템플릿을 설정하는 메소드
        
        Args:
            template: 새로운 프롬프트 템플릿
        """
        self.prompt_template = template
    
    def set_chunking_strategy(self, strategy: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """
        청크 분할 전략을 설정하는 메소드
        
        Args:
            strategy: 청크 분할 전략 ("simple", "paragraph", "sentence", "semantic")
            chunk_size: 청크 크기 (기본값: 1000)
            chunk_overlap: 청크 간 중첩 크기 (기본값: 200)
        """
        self.document_processor.chunking_strategy = strategy
        self.document_processor.chunk_size = chunk_size
        self.document_processor.chunk_overlap = chunk_overlap


# RAG 파이프라인 생성 함수
def create_pipeline(embedding_model_name: str = 'openai', 
                   query_model_name: str = 'gpt-3', 
                   vector_db: str = 'chromadb', 
                   api_key: str = None,
                   similarity: str = 'cosine', 
                   use_persistent_storage: bool = False,
                   temperature: float = 0.7,
                   **kwargs) -> RAGPipeline:
    """
    RAG 파이프라인을 생성하는 함수
    
    Args:
        embedding_model_name: 임베딩 모델 이름 (기본값: 'openai')
        query_model_name: 질의 모델 이름 (기본값: 'gpt-3')
        vector_db: 벡터 DB 이름 또는 인스턴스 (기본값: 'chromadb')
        api_key: OpenAI API 키 (기본값: None)
        similarity: 유사도 측정 방식 (기본값: 'cosine')
        use_persistent_storage: 영구 저장소 사용 여부 (기본값: False)
        temperature: 생성 다양성 조절 파라미터 (기본값: 0.7)
        **kwargs: 벡터 DB 초기화에 사용되는 추가 인자
        
    Returns:
        생성된 RAG 파이프라인
        
    Raises:
        ValueError: 지원하지 않는 모델인 경우
    """
    # API 키 확인
    if api_key is None:
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API 키가 필요합니다. API 키를 인자로 전달하거나 OPENAI_API_KEY 환경 변수를 설정하세요.")
    
    # 임베딩 모델 생성
    embedding_models = {
        'openai': lambda: OpenAIEmbedding(api_key)
    }
    
    # 질의 모델 생성
    query_models = {
        'gpt-3': lambda: OpenAIQuery(api_key, temperature=temperature),
        'gpt-4': lambda: OpenAIQuery(api_key, model="gpt-4", temperature=temperature)
    }
    
    # 플러그인에서 추가 모델 로드 시도
    try:
        from ..utils.plugin_manager import get_embedding_model_plugins, get_query_model_plugins
        embedding_models.update(get_embedding_model_plugins(api_key))
        query_models.update(get_query_model_plugins(api_key))
    except (ImportError, AttributeError):
        pass
    
    # 임베딩 모델 선택
    embedding_model_factory = embedding_models.get(embedding_model_name.lower())
    if embedding_model_factory is None:
        raise ValueError(f"지원하지 않는 임베딩 모델: {embedding_model_name}")
    
    # 질의 모델 선택
    query_model_factory = query_models.get(query_model_name.lower())
    if query_model_factory is None:
        raise ValueError(f"지원하지 않는 질의 모델: {query_model_name}")
    
    # 모델 인스턴스 생성
    embedding_model = embedding_model_factory()
    query_model = query_model_factory()
    
    # RAG 파이프라인 생성
    if isinstance(vector_db, str) and vector_db.lower() == 'chromadb':
        return RAGPipeline(
            embedding_model, 
            query_model, 
            vector_db, 
            similarity, 
            use_persistent_storage=use_persistent_storage, 
            **kwargs
        )
    else:
        return RAGPipeline(
            embedding_model, 
            query_model, 
            vector_db, 
            similarity, 
            **kwargs
        )

# RAG 파이프라인 실행 함수
def run_pipeline(pipeline: RAGPipeline, documents: List[str], query: str, 
                k: int = 1, threshold: Optional[float] = None,
                filter_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    RAG 파이프라인을 실행하는 함수
    
    Args:
        pipeline: RAG 파이프라인 인스턴스
        documents: 검색할 문서 리스트
        query: 사용자 쿼리
        k: 검색할 문서의 개수 (기본값: 1)
        threshold: 유사도 임계값 (기본값: None)
        filter_criteria: 검색 필터 조건 (기본값: None)
        
    Returns:
        검색 결과 (쿼리, 관련 문서, 유사도, 답변)
    """
    # 쿼리 임베딩 생성
    query_embedding = pipeline.embedding_model.embed([query])[0]
    
    # 관련 문서 검색
    if hasattr(pipeline.vector_db, 'search_with_filter') and filter_criteria:
        search_results = pipeline.vector_db.search_with_filter(
            query_embedding, k, threshold, filter_criteria
        )
    else:
        search_results = pipeline.vector_db.search(query_embedding, k, threshold)
    
    # 검색 결과 분리
    relevant_docs, distances = zip(*search_results) if search_results else ([], [])
    
    # 답변 생성
    answer = pipeline.query_model.generate(query, relevant_docs, pipeline.prompt_template)
    
    return {
        'query': query,
        'relevant_docs': relevant_docs,
        'distances': distances,
        'answer': answer
    }

# 문서 처리 함수
def process_documents(source: str, embedding_model: EmbeddingModel, 
                     chunking_strategy: str = "paragraph", 
                     chunk_size: int = 1000, 
                     chunk_overlap: int = 200,
                     recursive: bool = True) -> List[Tuple[str, np.ndarray, Dict[str, Any]]]:
    """
    문서를 처리하여 청크와 임베딩을 생성하는 함수
    
    Args:
        source: 파일 경로 또는 디렉토리 경로
        embedding_model: 임베딩 모델
        chunking_strategy: 청크 분할 전략 (기본값: "paragraph")
        chunk_size: 청크 크기 (기본값: 1000)
        chunk_overlap: 청크 간 중첩 크기 (기본값: 200)
        recursive: 하위 디렉토리를 재귀적으로 탐색할지 여부 (기본값: True)
        
    Returns:
        (청크 내용, 임베딩, 메타데이터) 튜플의 리스트
    """
    # 문서 처리기 생성
    document_processor = DocumentProcessor(
        chunking_strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    document_processor.set_embedding_model(embedding_model)
    
    # 파일 또는 디렉토리 처리
    if os.path.isfile(source):
        chunks = document_processor.process_file(source)
    elif os.path.isdir(source):
        chunks = document_processor.process_directory(source, recursive)
    else:
        raise ValueError(f"소스 '{source}'가 존재하지 않습니다.")
    
    # 청크 임베딩 생성
    contents = [chunk.content for chunk in chunks]
    embeddings = embedding_model.embed(contents)
    metadatas = [chunk.metadata for chunk in chunks]
    
    return list(zip(contents, embeddings, metadatas))