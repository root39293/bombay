import logging
import os
from typing import Dict, List, Optional, Union, Any
import numpy as np

from bombay.models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """임베딩 모델 추상 클래스"""
    
    def __init__(self):
        """임베딩 모델 초기화"""
        pass
    
    def get_embedding(self, text: str) -> List[float]:
        """텍스트의 임베딩 벡터 반환"""
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현해야 합니다.")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트의 임베딩 벡터 반환"""
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현해야 합니다.")
    
    def get_dimension(self) -> int:
        """임베딩 벡터의 차원 반환"""
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현해야 합니다.")
    
    def get_available_models(self) -> List[str]:
        """사용 가능한 모델 목록 반환"""
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현해야 합니다.")


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI 임베딩 모델 클래스"""
    
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, **kwargs):
        """
        OpenAI 임베딩 모델 초기화
        
        Args:
            model: 사용할 모델 이름 (기본값: 최신 모델)
            api_key: OpenAI API 키
            **kwargs: 추가 매개변수
        """
        super().__init__()
        
        # API 키 설정
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API 키가 필요합니다. api_key 매개변수를 통해 전달하거나 OPENAI_API_KEY 환경 변수를 설정하세요.")
        
        # 모델 레지스트리 초기화
        self.model_registry = ModelRegistry()
        
        # 모델 이름 설정
        if model is None or model.lower() == "openai" or model.lower() == "auto":
            # 최신 모델 사용
            self.model = self.model_registry.get_latest_embedding_model()
            logger.info(f"최신 OpenAI 임베딩 모델을 사용합니다: {self.model}")
        else:
            self.model = model
            logger.info(f"OpenAI 임베딩 모델을 사용합니다: {self.model}")
        
        # 임베딩 차원 설정
        self.dimension = self.model_registry.get_embedding_dimension(self.model)
        
        # OpenAI 클라이언트 초기화
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI 패키지가 설치되지 않았습니다. 'pip install openai>=1.0.0'를 실행하여 설치하세요.")
        
        # 추가 매개변수 설정
        self.kwargs = kwargs
    
    def get_embedding(self, text: str) -> List[float]:
        """
        텍스트의 임베딩 벡터 반환
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            임베딩 벡터
        """
        if not text or not text.strip():
            # 빈 텍스트는 0으로 채워진 벡터 반환
            return [0.0] * self.dimension
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                **self.kwargs
            )
            
            return response.data[0].embedding
        
        except Exception as e:
            logger.error(f"임베딩 생성 중 오류 발생: {e}")
            # 오류 발생 시 0으로 채워진 벡터 반환
            return [0.0] * self.dimension
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        여러 텍스트의 임베딩 벡터 반환
        
        Args:
            texts: 임베딩할 텍스트 목록
            
        Returns:
            임베딩 벡터 목록
        """
        if not texts:
            return []
        
        # 빈 텍스트 필터링 및 대체
        filtered_texts = []
        empty_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                filtered_texts.append(text)
            else:
                empty_indices.append(i)
        
        if not filtered_texts:
            # 모든 텍스트가 비어있으면 0으로 채워진 벡터 목록 반환
            return [[0.0] * self.dimension for _ in range(len(texts))]
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=filtered_texts,
                **self.kwargs
            )
            
            # 결과 정렬 (OpenAI API가 순서를 보장하지 않을 수 있음)
            embeddings = sorted(response.data, key=lambda x: x.index)
            result = [e.embedding for e in embeddings]
            
            # 빈 텍스트에 대한 0 벡터 삽입
            for idx in empty_indices:
                result.insert(idx, [0.0] * self.dimension)
            
            return result
        
        except Exception as e:
            logger.error(f"임베딩 생성 중 오류 발생: {e}")
            # 오류 발생 시 0으로 채워진 벡터 목록 반환
            return [[0.0] * self.dimension for _ in range(len(texts))]
    
    def get_dimension(self) -> int:
        """
        임베딩 벡터의 차원 반환
        
        Returns:
            임베딩 차원
        """
        return self.dimension
    
    def get_available_models(self) -> List[str]:
        """
        사용 가능한 OpenAI 임베딩 모델 목록 반환
        
        Returns:
            모델 이름 목록
        """
        # 모델 레지스트리에서 최신 정보 가져오기 시도
        self.model_registry.update_models_from_api(self.api_key)
        
        # 임베딩 모델 목록 반환
        return self.model_registry.get_all_embedding_models()
    
    def __str__(self) -> str:
        """모델 정보 문자열 반환"""
        return f"OpenAIEmbedding(model={self.model}, dimension={self.dimension})" 