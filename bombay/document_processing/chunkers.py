from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
import re
import logging
from bombay.document_processing.document import Document, Chunk
from bombay.models.embedding_model import EmbeddingModel
from bombay.document_processing.loaders import (
    TextLoader, MarkdownLoader, PDFLoader, DocxLoader, 
    HTMLLoader, CSVLoader, DirectoryLoader
)

logger = logging.getLogger(__name__)

class Chunker(ABC):
    """
    청크 분할기 추상 클래스
    """
    
    @abstractmethod
    def split(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Chunk]:
        """
        텍스트를 청크로 분할하는 메소드
        
        Args:
            text: 분할할 텍스트
            chunk_size: 청크 크기 (기본값: 1000)
            chunk_overlap: 청크 간 중첩 크기 (기본값: 200)
            
        Returns:
            청크 리스트
        """
        pass
    
    def split_document(self, document: Document, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Chunk]:
        """
        문서를 청크로 분할하는 메소드
        
        Args:
            document: 분할할 문서
            chunk_size: 청크 크기 (기본값: 1000)
            chunk_overlap: 청크 간 중첩 크기 (기본값: 200)
            
        Returns:
            청크 리스트
        """
        chunks = self.split(document.content, chunk_size, chunk_overlap)
        
        # 각 청크에 문서 메타데이터 추가
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                **document.metadata,
                "chunk_index": i,
                "chunk_count": len(chunks)
            })
        
        return chunks


class SimpleChunker(Chunker):
    """
    단순 청크 분할기
    """
    
    def split(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Chunk]:
        """
        텍스트를 단순히 크기 기반으로 분할하는 메소드
        
        Args:
            text: 분할할 텍스트
            chunk_size: 청크 크기 (기본값: 1000)
            chunk_overlap: 청크 간 중첩 크기 (기본값: 200)
            
        Returns:
            청크 리스트
        """
        chunks = []
        
        if not text:
            return chunks
        
        # 텍스트가 청크 크기보다 작은 경우
        if len(text) <= chunk_size:
            return [Chunk(text, {"method": "simple"})]
        
        # 청크 분할
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # 청크 생성
            chunk_text = text[start:end]
            chunks.append(Chunk(chunk_text, {
                "method": "simple",
                "start": start,
                "end": end
            }))
            
            # 다음 시작 위치 계산
            start = end - chunk_overlap
            
            # 중첩이 없는 경우
            if start >= end:
                start = end
        
        return chunks


class ParagraphChunker(Chunker):
    """
    단락 기반 청크 분할기
    """
    
    def split(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Chunk]:
        """
        텍스트를 단락 기반으로 분할하는 메소드
        
        Args:
            text: 분할할 텍스트
            chunk_size: 청크 크기 (기본값: 1000)
            chunk_overlap: 청크 간 중첩 크기 (기본값: 200)
            
        Returns:
            청크 리스트
        """
        chunks = []
        
        if not text:
            return chunks
        
        # 텍스트가 청크 크기보다 작은 경우
        if len(text) <= chunk_size:
            return [Chunk(text, {"method": "paragraph"})]
        
        # 단락 분할
        paragraphs = re.split(r"\n\s*\n", text)
        
        current_chunk = ""
        current_start = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 단락 추가 시 청크 크기를 초과하는 경우
            if len(current_chunk) + len(paragraph) + 2 > chunk_size and current_chunk:
                chunks.append(Chunk(current_chunk, {
                    "method": "paragraph",
                    "start": current_start,
                    "end": current_start + len(current_chunk)
                }))
                
                # 중첩 계산
                overlap_start = max(0, len(current_chunk) - chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
                current_start += overlap_start
            
            # 단락 추가
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
        
        # 마지막 청크 추가
        if current_chunk:
            chunks.append(Chunk(current_chunk, {
                "method": "paragraph",
                "start": current_start,
                "end": current_start + len(current_chunk)
            }))
        
        return chunks


class SentenceChunker(Chunker):
    """
    문장 기반 청크 분할기
    """
    
    def split(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Chunk]:
        """
        텍스트를 문장 기반으로 분할하는 메소드
        
        Args:
            text: 분할할 텍스트
            chunk_size: 청크 크기 (기본값: 1000)
            chunk_overlap: 청크 간 중첩 크기 (기본값: 200)
            
        Returns:
            청크 리스트
        """
        chunks = []
        
        if not text:
            return chunks
        
        # 텍스트가 청크 크기보다 작은 경우
        if len(text) <= chunk_size:
            return [Chunk(text, {"method": "sentence"})]
        
        # 문장 분할
        sentences = re.split(r"(?<=[.!?])\s+", text)
        
        current_chunk = ""
        current_sentences = []
        current_start = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 문장 추가 시 청크 크기를 초과하는 경우
            if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
                chunks.append(Chunk(current_chunk, {
                    "method": "sentence",
                    "start": current_start,
                    "end": current_start + len(current_chunk),
                    "sentence_count": len(current_sentences)
                }))
                
                # 중첩 계산
                overlap_sentences = []
                overlap_text = ""
                overlap_length = 0
                
                for s in reversed(current_sentences):
                    if overlap_length + len(s) + 1 <= chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_text = s + " " + overlap_text if overlap_text else s
                        overlap_length += len(s) + 1
                    else:
                        break
                
                current_chunk = overlap_text
                current_sentences = overlap_sentences
                current_start += len(current_chunk) - overlap_length
            
            # 문장 추가
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            
            current_sentences.append(sentence)
        
        # 마지막 청크 추가
        if current_chunk:
            chunks.append(Chunk(current_chunk, {
                "method": "sentence",
                "start": current_start,
                "end": current_start + len(current_chunk),
                "sentence_count": len(current_sentences)
            }))
        
        return chunks


class SemanticChunker(Chunker):
    """
    의미 기반 청크 분할기
    """
    
    def __init__(self, embedding_model=None):
        """
        의미 기반 청크 분할기 초기화
        
        Args:
            embedding_model: 임베딩 모델 (기본값: None)
        """
        self.embedding_model = embedding_model
        self.paragraph_chunker = ParagraphChunker()
    
    def split(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Chunk]:
        """
        텍스트를 의미 기반으로 분할하는 메소드
        
        Args:
            text: 분할할 텍스트
            chunk_size: 청크 크기 (기본값: 1000)
            chunk_overlap: 청크 간 중첩 크기 (기본값: 200)
            
        Returns:
            청크 리스트
        """
        # 임베딩 모델이 없는 경우 단락 기반 분할 사용
        if self.embedding_model is None:
            logger.warning("임베딩 모델이 없어 단락 기반 분할을 사용합니다.")
            return self.paragraph_chunker.split(text, chunk_size, chunk_overlap)
        
        try:
            import numpy as np
            from sklearn.cluster import AgglomerativeClustering
        except ImportError:
            logger.warning("의미 기반 분할을 위해 'pip install scikit-learn'을 실행하여 패키지를 설치하세요.")
            return self.paragraph_chunker.split(text, chunk_size, chunk_overlap)
        
        # 단락 분할
        paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        
        if not paragraphs:
            return []
        
        # 단락이 적은 경우
        if len(paragraphs) <= 1:
            return [Chunk(text, {"method": "semantic"})]
        
        # 임베딩 생성
        embeddings = self.embedding_model.embed(paragraphs)
        
        # 클러스터링
        n_clusters = max(1, min(len(paragraphs) // 3, len(paragraphs) // (chunk_size // 200)))
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = clustering.fit_predict(embeddings)
        
        # 클러스터별로 청크 생성
        chunks = []
        for cluster_id in range(n_clusters):
            cluster_paragraphs = [paragraphs[i] for i in range(len(paragraphs)) if clusters[i] == cluster_id]
            cluster_text = "\n\n".join(cluster_paragraphs)
            
            # 청크 크기 초과 시 추가 분할
            if len(cluster_text) > chunk_size:
                sub_chunks = self.paragraph_chunker.split(cluster_text, chunk_size, chunk_overlap)
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata["method"] = "semantic_paragraph"
                    sub_chunk.metadata["cluster_id"] = cluster_id
                chunks.extend(sub_chunks)
            else:
                chunks.append(Chunk(cluster_text, {
                    "method": "semantic",
                    "cluster_id": cluster_id
                }))
        
        return chunks


class DocumentProcessor:
    """
    문서 처리기
    """
    
    def __init__(self, chunking_strategy: str = "paragraph", chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        문서 처리기 초기화
        
        Args:
            chunking_strategy: 청크 분할 전략 (기본값: "paragraph")
            chunk_size: 청크 크기 (기본값: 1000)
            chunk_overlap: 청크 간 중첩 크기 (기본값: 200)
        """
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.loaders = {
            "txt": TextLoader(),
            "md": MarkdownLoader(),
            "pdf": PDFLoader(),
            "docx": DocxLoader(),
            "html": HTMLLoader(),
            "htm": HTMLLoader(),
            "csv": CSVLoader()
        }
        self.chunkers = {
            "simple": SimpleChunker(),
            "paragraph": ParagraphChunker(),
            "sentence": SentenceChunker()
        }
    
    def set_embedding_model(self, embedding_model):
        """
        임베딩 모델을 설정하는 메소드
        
        Args:
            embedding_model: 임베딩 모델
        """
        self.chunkers["semantic"] = SemanticChunker(embedding_model)
    
    def process_file(self, file_path: str) -> List[Chunk]:
        """
        파일을 처리하는 메소드
        
        Args:
            file_path: 파일 경로
            
        Returns:
            청크 리스트
        """
        # 파일 확장자 확인
        extension = file_path.split(".")[-1].lower()
        
        if extension not in self.loaders:
            raise ValueError(f"지원하지 않는 파일 형식: {extension}")
        
        # 문서 로드
        document = self.loaders[extension].load(file_path)
        
        # 청크 분할
        chunker = self.chunkers.get(self.chunking_strategy, self.chunkers["paragraph"])
        chunks = chunker.split_document(document, self.chunk_size, self.chunk_overlap)
        
        return chunks
    
    def process_directory(self, directory: str, recursive: bool = True) -> List[Chunk]:
        """
        디렉토리를 처리하는 메소드
        
        Args:
            directory: 디렉토리 경로
            recursive: 하위 디렉토리를 재귀적으로 탐색할지 여부 (기본값: True)
            
        Returns:
            청크 리스트
        """
        directory_loader = DirectoryLoader(recursive=recursive)
        documents = directory_loader.load(directory)
        
        chunks = []
        for document in documents:
            chunker = self.chunkers.get(self.chunking_strategy, self.chunkers["paragraph"])
            document_chunks = chunker.split_document(document, self.chunk_size, self.chunk_overlap)
            chunks.extend(document_chunks)
        
        return chunks
    
    def register_loader(self, extension: str, loader):
        """
        로더를 등록하는 메소드
        
        Args:
            extension: 파일 확장자
            loader: 문서 로더
        """
        self.loaders[extension.lower()] = loader
    
    def register_chunker(self, name: str, chunker):
        """
        청크 분할기를 등록하는 메소드
        
        Args:
            name: 청크 분할기 이름
            chunker: 청크 분할기
        """
        self.chunkers[name.lower()] = chunker 