"""
문서 처리 모듈의 기본 클래스 정의
"""

import logging
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

class Document:
    """문서 클래스"""
    
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        문서 초기화
        
        Args:
            content: 문서 내용
            metadata: 문서 메타데이터 (기본값: None)
        """
        self.content = content
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        """문자열 표현"""
        return f"Document(content={self.content[:50]}..., metadata={self.metadata})"


class Chunk:
    """청크 클래스"""
    
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        청크 초기화
        
        Args:
            content: 청크 내용
            metadata: 청크 메타데이터 (기본값: None)
        """
        self.content = content
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        """문자열 표현"""
        return f"Chunk(content={self.content[:50]}..., metadata={self.metadata})" 