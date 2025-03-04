"""
ChromaDB 벡터 데이터베이스 모듈
"""

import logging
import os
import uuid
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError("ChromaDB 패키지가 설치되지 않았습니다. 'pip install chromadb>=0.4.0'를 실행하여 설치하세요.")

from bombay.vector_db.vector_db import VectorDB

logger = logging.getLogger(__name__)

class ChromaDB(VectorDB):
    """ChromaDB 벡터 데이터베이스 클래스"""
    
    def __init__(self, 
                 dim: int = 1536, 
                 collection_name: str = "bombay_collection",
                 use_persistent_storage: bool = False,
                 persist_directory: str = "./chroma_db",
                 similarity: str = "cosine",
                 **kwargs):
        """
        ChromaDB 벡터 데이터베이스 초기화
        
        Args:
            dim: 벡터 차원
            collection_name: 컬렉션 이름
            use_persistent_storage: 영구 저장소 사용 여부
            persist_directory: 영구 저장소 디렉토리
            similarity: 유사도 측정 방법 ('cosine', 'l2', 'ip')
            **kwargs: 추가 매개변수
        """
        super().__init__(dim=dim, **kwargs)
        
        # 설정
        self.collection_name = collection_name
        self.use_persistent_storage = use_persistent_storage
        self.persist_directory = persist_directory
        self.similarity = similarity
        
        # ChromaDB 클라이언트 초기화
        self._init_client()
        
        # 컬렉션 초기화
        self._init_collection()
    
    def _init_client(self) -> None:
        """ChromaDB 클라이언트 초기화"""
        try:
            if self.use_persistent_storage:
                # 영구 저장소 사용
                os.makedirs(self.persist_directory, exist_ok=True)
                self.client = chromadb.PersistentClient(path=self.persist_directory)
                logger.info(f"ChromaDB 영구 저장소를 {self.persist_directory}에 초기화했습니다.")
            else:
                # 인메모리 클라이언트 사용
                self.client = chromadb.Client(Settings(anonymized_telemetry=False))
                logger.info("ChromaDB 인메모리 클라이언트를 초기화했습니다.")
        
        except Exception as e:
            logger.error(f"ChromaDB 클라이언트 초기화 중 오류 발생: {e}")
            raise
    
    def _init_collection(self) -> None:
        """컬렉션 초기화"""
        try:
            # 기존 컬렉션 가져오기 또는 새로 생성
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=None  # 임베딩 함수는 외부에서 처리
                )
                logger.info(f"기존 ChromaDB 컬렉션 '{self.collection_name}'을 로드했습니다.")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=None,  # 임베딩 함수는 외부에서 처리
                    metadata={"hnsw:space": self.similarity}
                )
                logger.info(f"새 ChromaDB 컬렉션 '{self.collection_name}'을 생성했습니다.")
        
        except Exception as e:
            logger.error(f"ChromaDB 컬렉션 초기화 중 오류 발생: {e}")
            raise
    
    def add(self, 
            vectors: List[List[float]], 
            documents: List[str], 
            metadatas: Optional[List[Dict[str, Any]]] = None, 
            ids: Optional[List[str]] = None) -> List[str]:
        """
        벡터 추가
        
        Args:
            vectors: 벡터 목록
            documents: 문서 목록
            metadatas: 메타데이터 목록
            ids: ID 목록
            
        Returns:
            추가된 문서의 ID 목록
        """
        if not vectors or not documents:
            return []
        
        # 입력 검증
        if len(vectors) != len(documents):
            raise ValueError(f"벡터 수({len(vectors)})와 문서 수({len(documents)})가 일치하지 않습니다.")
        
        # 메타데이터 확인
        if metadatas is None:
            metadatas = [{} for _ in range(len(vectors))]
        elif len(metadatas) != len(vectors):
            raise ValueError(f"벡터 수({len(vectors)})와 메타데이터 수({len(metadatas)})가 일치하지 않습니다.")
        
        # ID 확인
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        elif len(ids) != len(vectors):
            raise ValueError(f"벡터 수({len(vectors)})와 ID 수({len(ids)})가 일치하지 않습니다.")
        
        try:
            # 문서 추가
            self.collection.add(
                embeddings=vectors,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            return ids
        
        except Exception as e:
            logger.error(f"벡터 추가 중 오류 발생: {e}")
            return []
    
    def search(self, 
               query_vector: List[float], 
               k: int = 4, 
               filter_criteria: Optional[Dict[str, Any]] = None) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """
        벡터 검색
        
        Args:
            query_vector: 쿼리 벡터
            k: 검색할 문서 수
            filter_criteria: 필터링 기준
            
        Returns:
            (문서 목록, 거리 목록, 메타데이터 목록) 튜플
        """
        try:
            # 필터 쿼리 생성
            where = filter_criteria if filter_criteria else None
            
            # 검색 수행
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=k,
                where=where
            )
            
            # 결과 추출
            if not results["documents"] or not results["documents"][0]:
                return [], [], []
            
            documents = results["documents"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(documents)
            metadatas = results["metadatas"][0] if "metadatas" in results else [{} for _ in range(len(documents))]
            
            return documents, distances, metadatas
        
        except Exception as e:
            logger.error(f"벡터 검색 중 오류 발생: {e}")
            return [], [], []
    
    def delete(self, ids: List[str]) -> None:
        """
        벡터 삭제
        
        Args:
            ids: 삭제할 문서의 ID 목록
        """
        if not ids:
            return
        
        try:
            # 문서 삭제
            self.collection.delete(ids=ids)
        
        except Exception as e:
            logger.error(f"벡터 삭제 중 오류 발생: {e}")
    
    def clear(self) -> None:
        """모든 벡터 삭제"""
        try:
            # 컬렉션 삭제 후 재생성
            self.client.delete_collection(self.collection_name)
            self._init_collection()
        
        except Exception as e:
            logger.error(f"컬렉션 초기화 중 오류 발생: {e}")
    
    def count(self) -> int:
        """
        벡터 수 반환
        
        Returns:
            벡터 수
        """
        try:
            return self.collection.count()
        
        except Exception as e:
            logger.error(f"벡터 수 조회 중 오류 발생: {e}")
            return 0
    
    def save(self, path: str = None) -> None:
        """
        벡터 데이터베이스 저장
        
        Args:
            path: 저장 경로 (영구 저장소 사용 시 무시됨)
        """
        if self.use_persistent_storage:
            # 영구 저장소 사용 시 자동 저장
            try:
                self.client.persist()
                logger.info(f"ChromaDB 데이터를 {self.persist_directory}에 저장했습니다.")
            
            except Exception as e:
                logger.error(f"ChromaDB 데이터 저장 중 오류 발생: {e}")
        else:
            logger.warning("영구 저장소를 사용하지 않아 저장할 수 없습니다. use_persistent_storage=True로 설정하세요.")
    
    def load(self, path: str = None) -> None:
        """
        벡터 데이터베이스 로드
        
        Args:
            path: 로드 경로 (영구 저장소 사용 시 무시됨)
        """
        if self.use_persistent_storage:
            # 영구 저장소 사용 시 재초기화
            self._init_client()
            self._init_collection()
            logger.info(f"ChromaDB 데이터를 {self.persist_directory}에서 로드했습니다.")
        else:
            logger.warning("영구 저장소를 사용하지 않아 로드할 수 없습니다. use_persistent_storage=True로 설정하세요.")
    
    def __str__(self) -> str:
        """문자열 표현"""
        storage_type = "영구 저장소" if self.use_persistent_storage else "인메모리"
        return f"ChromaDB(collection={self.collection_name}, storage={storage_type}, count={self.count()})" 