"""
문서 처리 모듈

이 패키지는 문서 로딩, 청킹, 처리를 위한 클래스를 제공합니다.
"""

from bombay.document_processing.document import Document, Chunk
from bombay.document_processing.loaders import (
    DocumentLoader, TextLoader, MarkdownLoader, PDFLoader, 
    DocxLoader, HTMLLoader, CSVLoader, DirectoryLoader
)
from bombay.document_processing.chunkers import Chunker

class DocumentProcessor:
    """문서 처리 클래스"""
    
    def __init__(self, embedding_model=None, chunker=None):
        """
        문서 처리기 초기화
        
        Args:
            embedding_model: 임베딩 모델 (기본값: None)
            chunker: 청크 분할기 (기본값: None)
        """
        self.embedding_model = embedding_model
        self.chunker = chunker
    
    def process_document(self, document, chunk_size=1000, chunk_overlap=200):
        """
        문서 처리
        
        Args:
            document: 처리할 문서
            chunk_size: 청크 크기
            chunk_overlap: 청크 중첩 크기
            
        Returns:
            처리된 청크 목록
        """
        if self.chunker:
            return self.chunker.split_document(document, chunk_size, chunk_overlap)
        return [Chunk(document.content, document.metadata)]
    
    def process_text(self, text, metadata=None, chunk_size=1000, chunk_overlap=200):
        """
        텍스트 처리
        
        Args:
            text: 처리할 텍스트
            metadata: 메타데이터
            chunk_size: 청크 크기
            chunk_overlap: 청크 중첩 크기
            
        Returns:
            처리된 청크 목록
        """
        document = Document(text, metadata or {})
        return self.process_document(document, chunk_size, chunk_overlap)

__all__ = [
    'Document', 'Chunk', 'DocumentProcessor',
    'DocumentLoader', 'TextLoader', 'MarkdownLoader', 'PDFLoader', 
    'DocxLoader', 'HTMLLoader', 'CSVLoader', 'DirectoryLoader',
    'Chunker'
] 