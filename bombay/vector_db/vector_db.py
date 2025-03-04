"""
벡터 데이터베이스 기본 클래스 모듈
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple

logger = logging.getLogger(__name__)

class VectorDB:
    """벡터 데이터베이스 추상 클래스"""
    
    def __init__(self, dim: int = 1536, **kwargs):
        """
        벡터 데이터베이스 초기화
        
        Args:
            dim: 벡터 차원
            **kwargs: 추가 매개변수
        """
        self.dim = dim
    
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
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현해야 합니다.")
    
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
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현해야 합니다.")
    
    def delete(self, ids: List[str]) -> None:
        """
        벡터 삭제
        
        Args:
            ids: 삭제할 문서의 ID 목록
        """
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현해야 합니다.")
    
    def clear(self) -> None:
        """모든 벡터 삭제"""
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현해야 합니다.")
    
    def count(self) -> int:
        """
        벡터 수 반환
        
        Returns:
            벡터 수
        """
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현해야 합니다.")
    
    def save(self, path: str) -> None:
        """
        벡터 데이터베이스 저장
        
        Args:
            path: 저장 경로
        """
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현해야 합니다.")
    
    def load(self, path: str) -> None:
        """
        벡터 데이터베이스 로드
        
        Args:
            path: 로드 경로
        """
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현해야 합니다.")
    
    def __len__(self) -> int:
        """벡터 수 반환"""
        return self.count() 