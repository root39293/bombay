"""
Pinecone 벡터 데이터베이스 플러그인 모듈
"""

import logging
import os
import uuid
import json
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np

# 벡터 데이터베이스 기본 클래스 임포트
from bombay.vector_db.vector_db import VectorDB

logger = logging.getLogger(__name__)

class PineconeDB(VectorDB):
    """Pinecone 벡터 데이터베이스 클래스"""
    
    def __init__(self, 
                 dim: int = 1536, 
                 api_key: Optional[str] = None,
                 api_key_pinecone: Optional[str] = None,
                 environment: str = "us-west1-gcp",
                 index_name: str = "bombay-index",
                 namespace: str = "default",
                 create_index: bool = False,
                 similarity: str = "cosine",
                 **kwargs):
        """
        Pinecone 벡터 데이터베이스 초기화
        
        Args:
            dim: 벡터 차원
            api_key: OpenAI API 키 (사용하지 않음)
            api_key_pinecone: Pinecone API 키
            environment: Pinecone 환경
            index_name: 인덱스 이름
            namespace: 네임스페이스
            create_index: 인덱스가 없을 경우 생성 여부
            similarity: 유사도 측정 방법 ('cosine', 'dotproduct', 'euclidean')
            **kwargs: 추가 매개변수
        """
        super().__init__(dim=dim, **kwargs)
        
        # Pinecone 설정
        self.api_key = api_key_pinecone or os.environ.get("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("Pinecone API 키가 필요합니다. api_key_pinecone 매개변수를 통해 전달하거나 PINECONE_API_KEY 환경 변수를 설정하세요.")
        
        self.environment = environment
        self.index_name = index_name
        self.namespace = namespace
        self.create_index = create_index
        
        # 유사도 매핑
        similarity_map = {
            "cosine": "cosine",
            "dotproduct": "dotproduct",
            "dot": "dotproduct",
            "euclidean": "euclidean",
            "l2": "euclidean"
        }
        self.similarity = similarity_map.get(similarity.lower(), "cosine")
        
        # Pinecone 초기화
        self._init_pinecone()
    
    def _init_pinecone(self) -> None:
        """Pinecone 초기화"""
        try:
            import pinecone
            
            # Pinecone 초기화
            pinecone.init(api_key=self.api_key, environment=self.environment)
            
            # 인덱스 확인
            if self.index_name not in pinecone.list_indexes():
                if self.create_index:
                    # 인덱스 생성
                    pinecone.create_index(
                        name=self.index_name,
                        dimension=self.dim,
                        metric=self.similarity
                    )
                    logger.info(f"Pinecone 인덱스 '{self.index_name}'을 생성했습니다.")
                else:
                    raise ValueError(f"Pinecone 인덱스 '{self.index_name}'이 존재하지 않습니다. create_index=True로 설정하여 자동 생성하세요.")
            
            # 인덱스 연결
            self.index = pinecone.Index(self.index_name)
            logger.info(f"Pinecone 인덱스 '{self.index_name}'에 연결했습니다.")
        
        except ImportError:
            raise ImportError("Pinecone 패키지가 설치되지 않았습니다. 'pip install pinecone-client>=2.2.0'를 실행하여 설치하세요.")
        
        except Exception as e:
            logger.error(f"Pinecone 초기화 중 오류 발생: {e}")
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
            # 문서를 메타데이터에 추가
            for i, doc in enumerate(documents):
                metadatas[i]["text"] = doc
            
            # 벡터 추가 (배치 처리)
            batch_size = 100  # Pinecone 권장 배치 크기
            for i in range(0, len(vectors), batch_size):
                batch_vectors = vectors[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]
                
                # 벡터 데이터 준비
                items = []
                for j, vec in enumerate(batch_vectors):
                    items.append({
                        "id": batch_ids[j],
                        "values": vec,
                        "metadata": batch_metadatas[j]
                    })
                
                # 업서트 수행
                self.index.upsert(vectors=items, namespace=self.namespace)
            
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
            filter_dict = filter_criteria if filter_criteria else None
            
            # 검색 수행
            results = self.index.query(
                vector=query_vector,
                top_k=k,
                namespace=self.namespace,
                filter=filter_dict,
                include_metadata=True
            )
            
            # 결과 추출
            documents = []
            distances = []
            metadatas = []
            
            for match in results.matches:
                # 문서 추출 (메타데이터에서)
                doc = match.metadata.get("text", "")
                documents.append(doc)
                
                # 거리 추출
                distances.append(match.score)
                
                # 메타데이터 추출 (텍스트 제외)
                metadata = {k: v for k, v in match.metadata.items() if k != "text"}
                metadatas.append(metadata)
            
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
            # 벡터 삭제
            self.index.delete(ids=ids, namespace=self.namespace)
        
        except Exception as e:
            logger.error(f"벡터 삭제 중 오류 발생: {e}")
    
    def clear(self) -> None:
        """모든 벡터 삭제"""
        try:
            # 네임스페이스 삭제
            self.index.delete(delete_all=True, namespace=self.namespace)
            logger.info(f"Pinecone 네임스페이스 '{self.namespace}'의 모든 벡터를 삭제했습니다.")
        
        except Exception as e:
            logger.error(f"벡터 삭제 중 오류 발생: {e}")
    
    def count(self) -> int:
        """
        벡터 수 반환
        
        Returns:
            벡터 수
        """
        try:
            # 통계 조회
            stats = self.index.describe_index_stats()
            
            # 네임스페이스별 벡터 수
            namespaces = stats.get("namespaces", {})
            namespace_stats = namespaces.get(self.namespace, {})
            
            return namespace_stats.get("vector_count", 0)
        
        except Exception as e:
            logger.error(f"벡터 수 조회 중 오류 발생: {e}")
            return 0
    
    def save(self, path: str) -> None:
        """
        벡터 데이터베이스 저장 (지원하지 않음)
        
        Args:
            path: 저장 경로
        """
        logger.warning("Pinecone은 로컬 저장을 지원하지 않습니다. 데이터는 Pinecone 클라우드에 저장됩니다.")
    
    def load(self, path: str) -> None:
        """
        벡터 데이터베이스 로드 (지원하지 않음)
        
        Args:
            path: 로드 경로
        """
        logger.warning("Pinecone은 로컬 로드를 지원하지 않습니다. 데이터는 Pinecone 클라우드에서 로드됩니다.")
    
    def __str__(self) -> str:
        """문자열 표현"""
        return f"PineconeDB(index={self.index_name}, namespace={self.namespace}, count={self.count()})" 