# bombay/pipeline/embedding_models.py
from abc import ABC, abstractmethod
from openai import OpenAI
import numpy as np
from typing import List, Optional, Union
from ..utils.model_registry import ModelRegistry

class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트를 임베딩하는 메소드
        
        Args:
            texts: 임베딩할 텍스트 리스트
            
        Returns:
            임베딩 벡터 리스트
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        임베딩 벡터의 차원을 반환하는 메소드
        
        Returns:
            임베딩 벡터의 차원
        """
        pass


# OpenAI 임베딩 모델 어댑터
class OpenAIEmbedding(EmbeddingModel):
    def __init__(self, api_key: str, model: Optional[str] = None):
        """
        OpenAI 임베딩 모델 초기화
        
        Args:
            api_key: OpenAI API 키
            model: 사용할 OpenAI 임베딩 모델 (기본값: None, 자동 선택)
        """
        self.client = OpenAI(api_key=api_key)
        self.model_registry = ModelRegistry(api_key)
        
        # 모델이 지정되지 않은 경우 권장 모델 사용
        if model is None:
            self.model = self.model_registry.get_recommended_model("embedding")
        else:
            self.model = model
            
        self.dimension = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트를 OpenAI 임베딩 모델로 임베딩하는 메소드
        
        Args:
            texts: 임베딩할 텍스트 리스트
            
        Returns:
            임베딩 리스트
        """
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        embeddings = [data.embedding for data in response.data]
        if self.dimension is None:
            self.dimension = len(embeddings[0])
        return embeddings

    def get_dimension(self) -> int:
        """
        OpenAI 임베딩 모델의 임베딩 차원을 반환하는 메소드
        
        Returns:
            임베딩의 차원
        """
        if self.dimension is None:
            sample_document = 'This is a sample document to get embedding dimension.'
            self.dimension = len(self.embed([sample_document])[0])
        return self.dimension
    
    def get_available_models(self) -> List[str]:
        """
        사용 가능한 임베딩 모델 목록을 반환하는 메소드
        
        Returns:
            사용 가능한 임베딩 모델 목록
        """
        return self.model_registry.get_available_models()["embedding"]
    
    def update_model(self, model: str) -> None:
        """
        사용할 임베딩 모델을 업데이트하는 메소드
        
        Args:
            model: 새로운 임베딩 모델 이름
        """
        self.model = model
        self.dimension = None  # 차원 정보 초기화