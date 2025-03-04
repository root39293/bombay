"""
HNSWLib 벡터 데이터베이스 모듈
"""

import logging
import os
import json
import uuid
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np

try:
    import hnswlib
except ImportError:
    raise ImportError("HNSWLib 패키지가 설치되지 않았습니다. 'pip install hnswlib>=0.7.0'를 실행하여 설치하세요.")

from bombay.vector_db.vector_db import VectorDB

logger = logging.getLogger(__name__)

class HNSWLib(VectorDB):
    """HNSWLib 벡터 데이터베이스 클래스"""
    
    def __init__(self, 
                 dim: int = 1536, 
                 space: str = 'cosine', 
                 max_elements: int = 10000, 
                 ef_construction: int = 200, 
                 M: int = 16, 
                 **kwargs):
        """
        HNSWLib 벡터 데이터베이스 초기화
        
        Args:
            dim: 벡터 차원
            space: 유사도 공간 ('cosine', 'l2', 'ip')
            max_elements: 최대 요소 수
            ef_construction: 인덱스 구성 매개변수
            M: 인덱스 구성 매개변수
            **kwargs: 추가 매개변수
        """
        super().__init__(dim=dim, **kwargs)
        
        # 유사도 공간 설정
        self.space = space
        
        # 인덱스 매개변수 설정
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.M = M
        
        # 인덱스 초기화
        self._init_index()
        
        # 문서 및 메타데이터 저장소
        self.documents = {}
        self.metadatas = {}
        
        # 현재 요소 수
        self.current_count = 0
    
    def _init_index(self) -> None:
        """인덱스 초기화"""
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(
            max_elements=self.max_elements,
            ef_construction=self.ef_construction,
            M=self.M
        )
        self.index.set_ef(50)  # 검색 매개변수
    
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
        
        # 인덱스 크기 확인 및 조정
        if self.current_count + len(vectors) > self.max_elements:
            new_size = max(self.max_elements * 2, self.current_count + len(vectors))
            logger.info(f"인덱스 크기를 {self.max_elements}에서 {new_size}로 조정합니다.")
            self.max_elements = new_size
            
            # 기존 데이터 백업
            old_documents = self.documents.copy()
            old_metadatas = self.metadatas.copy()
            old_count = self.current_count
            
            # 인덱스 재초기화
            self._init_index()
            self.index.resize_index(self.max_elements)
            
            # 기존 데이터 복원
            if old_count > 0:
                old_ids = list(old_documents.keys())
                old_vectors = [self.index.get_items(i)[0] for i in range(old_count)]
                old_docs = [old_documents[id] for id in old_ids]
                old_metas = [old_metadatas[id] for id in old_ids]
                
                self.index.add_items(
                    np.array(old_vectors), 
                    np.arange(old_count)
                )
                
                for i, id in enumerate(old_ids):
                    self.documents[id] = old_docs[i]
                    self.metadatas[id] = old_metas[i]
        
        # 벡터 추가
        try:
            self.index.add_items(
                np.array(vectors), 
                np.arange(self.current_count, self.current_count + len(vectors))
            )
            
            # 문서 및 메타데이터 저장
            for i, id in enumerate(ids):
                self.documents[id] = documents[i]
                self.metadatas[id] = metadatas[i]
            
            # 현재 요소 수 업데이트
            self.current_count += len(vectors)
            
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
        if self.current_count == 0:
            return [], [], []
        
        # k 값 조정
        k = min(k, self.current_count)
        
        # 필터링이 필요한 경우 더 많은 결과 검색
        if filter_criteria:
            search_k = min(k * 10, self.current_count)
        else:
            search_k = k
        
        try:
            # 검색 수행
            labels, distances = self.index.knn_query(np.array([query_vector]), k=search_k)
            
            # 결과 변환
            labels = labels[0]
            distances = distances[0]
            
            # ID 목록 생성
            ids = list(self.documents.keys())
            result_ids = [ids[label] for label in labels]
            
            # 문서 및 메타데이터 목록 생성
            result_docs = [self.documents[id] for id in result_ids]
            result_metas = [self.metadatas[id] for id in result_ids]
            
            # 필터링 적용
            if filter_criteria:
                filtered_results = []
                
                for i, meta in enumerate(result_metas):
                    match = True
                    
                    for key, value in filter_criteria.items():
                        if key not in meta or meta[key] != value:
                            match = False
                            break
                    
                    if match:
                        filtered_results.append((result_docs[i], distances[i], result_metas[i], result_ids[i]))
                
                # 결과 정렬 및 제한
                filtered_results = sorted(filtered_results, key=lambda x: x[1])[:k]
                
                if not filtered_results:
                    return [], [], []
                
                result_docs, distances, result_metas, result_ids = zip(*filtered_results)
            
            return result_docs, distances.tolist(), result_metas
        
        except Exception as e:
            logger.error(f"벡터 검색 중 오류 발생: {e}")
            return [], [], []
    
    def delete(self, ids: List[str]) -> None:
        """
        벡터 삭제
        
        Args:
            ids: 삭제할 문서의 ID 목록
        """
        # HNSWLib는 삭제를 직접 지원하지 않으므로 재구성 필요
        if not ids:
            return
        
        # 삭제할 ID 필터링
        valid_ids = [id for id in ids if id in self.documents]
        
        if not valid_ids:
            return
        
        # 남길 ID 목록 생성
        remaining_ids = [id for id in self.documents.keys() if id not in valid_ids]
        
        if not remaining_ids:
            # 모든 문서 삭제
            self.clear()
            return
        
        # 남길 문서 및 메타데이터 백업
        remaining_docs = [self.documents[id] for id in remaining_ids]
        remaining_metas = [self.metadatas[id] for id in remaining_ids]
        
        # 인덱스 재초기화
        self._init_index()
        
        # 문서 및 메타데이터 초기화
        self.documents = {}
        self.metadatas = {}
        self.current_count = 0
        
        # 남은 문서 다시 추가
        if remaining_docs:
            # 벡터 다시 계산 필요 (HNSWLib는 벡터를 직접 저장하지 않음)
            logger.warning("HNSWLib는 벡터를 직접 저장하지 않아 삭제 후 재구성이 필요합니다.")
    
    def clear(self) -> None:
        """모든 벡터 삭제"""
        self._init_index()
        self.documents = {}
        self.metadatas = {}
        self.current_count = 0
    
    def count(self) -> int:
        """
        벡터 수 반환
        
        Returns:
            벡터 수
        """
        return self.current_count
    
    def save(self, path: str) -> None:
        """
        벡터 데이터베이스 저장
        
        Args:
            path: 저장 경로
        """
        try:
            # 디렉토리 생성
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # 인덱스 저장
            self.index.save_index(f"{path}.index")
            
            # 문서 및 메타데이터 저장
            with open(f"{path}.json", 'w') as f:
                json.dump({
                    'documents': self.documents,
                    'metadatas': self.metadatas,
                    'current_count': self.current_count,
                    'dim': self.dim,
                    'space': self.space,
                    'max_elements': self.max_elements,
                    'ef_construction': self.ef_construction,
                    'M': self.M
                }, f)
            
            logger.info(f"벡터 데이터베이스를 {path}에 저장했습니다.")
        
        except Exception as e:
            logger.error(f"벡터 데이터베이스 저장 중 오류 발생: {e}")
    
    def load(self, path: str) -> None:
        """
        벡터 데이터베이스 로드
        
        Args:
            path: 로드 경로
        """
        try:
            # 문서 및 메타데이터 로드
            with open(f"{path}.json", 'r') as f:
                data = json.load(f)
            
            self.documents = data['documents']
            self.metadatas = data['metadatas']
            self.current_count = data['current_count']
            self.dim = data['dim']
            self.space = data['space']
            self.max_elements = data['max_elements']
            self.ef_construction = data['ef_construction']
            self.M = data['M']
            
            # 인덱스 초기화
            self._init_index()
            
            # 인덱스 로드
            self.index.load_index(f"{path}.index", max_elements=self.max_elements)
            
            logger.info(f"벡터 데이터베이스를 {path}에서 로드했습니다.")
        
        except Exception as e:
            logger.error(f"벡터 데이터베이스 로드 중 오류 발생: {e}")
    
    def __str__(self) -> str:
        """문자열 표현"""
        return f"HNSWLib(dim={self.dim}, space={self.space}, count={self.current_count})" 