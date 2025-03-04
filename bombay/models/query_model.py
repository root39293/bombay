"""
질의 모델 모듈
"""

import logging
import os
from typing import Dict, List, Optional, Union, Any
import json

from bombay.models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class QueryModel:
    """질의 모델 추상 클래스"""
    
    def __init__(self):
        """질의 모델 초기화"""
        pass
    
    def generate(self, query: str, context: Optional[List[str]] = None, **kwargs) -> str:
        """
        응답 생성
        
        Args:
            query: 질의 텍스트
            context: 컨텍스트 문서 목록 (기본값: None)
            **kwargs: 추가 매개변수
            
        Returns:
            생성된 응답
        """
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현해야 합니다.")
    
    def get_available_models(self) -> List[str]:
        """
        사용 가능한 모델 목록 반환
        
        Returns:
            모델 이름 목록
        """
        raise NotImplementedError("이 메서드는 하위 클래스에서 구현해야 합니다.")


class OpenAIQuery(QueryModel):
    """OpenAI 질의 모델 클래스"""
    
    def __init__(self, 
                 model: Optional[str] = None, 
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 **kwargs):
        """
        OpenAI 질의 모델 초기화
        
        Args:
            model: 사용할 모델 이름 (기본값: 최신 모델)
            api_key: OpenAI API 키
            temperature: 온도 (기본값: 0.7)
            max_tokens: 최대 토큰 수 (기본값: None)
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
        if model is None or model.lower() == "auto":
            # 최신 GPT-3 모델 사용
            self.model = self.model_registry.get_latest_query_model("gpt-3")
            logger.info(f"최신 GPT-3 모델을 사용합니다: {self.model}")
        elif model.lower() in ["gpt-3", "gpt3"]:
            # 최신 GPT-3 모델 사용
            self.model = self.model_registry.get_latest_query_model("gpt-3")
            logger.info(f"최신 GPT-3 모델을 사용합니다: {self.model}")
        elif model.lower() in ["gpt-4", "gpt4"]:
            # 최신 GPT-4 모델 사용
            self.model = self.model_registry.get_latest_query_model("gpt-4")
            logger.info(f"최신 GPT-4 모델을 사용합니다: {self.model}")
        elif model.lower() == "reasoning":
            # 최신 추론 모델 사용
            self.model = self.model_registry.get_latest_query_model("reasoning")
            logger.info(f"최신 추론 모델을 사용합니다: {self.model}")
        else:
            self.model = model
            logger.info(f"OpenAI 질의 모델을 사용합니다: {self.model}")
        
        # 모델 별칭 해결
        self.model = self.model_registry.resolve_model_alias(self.model)
        
        # 온도 설정
        self.temperature = temperature
        
        # 최대 토큰 수 설정
        if max_tokens is None:
            # 모델의 최대 출력 토큰 수 사용
            self.max_tokens = self.model_registry.get_max_output_tokens(self.model)
        else:
            self.max_tokens = max_tokens
        
        # OpenAI 클라이언트 초기화
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI 패키지가 설치되지 않았습니다. 'pip install openai>=1.0.0'를 실행하여 설치하세요.")
        
        # 추가 매개변수 설정
        self.kwargs = kwargs
    
    def _create_prompt(self, query: str, context: Optional[List[str]] = None) -> str:
        """
        프롬프트 생성
        
        Args:
            query: 질의 텍스트
            context: 컨텍스트 문서 목록
            
        Returns:
            프롬프트 텍스트
        """
        if not context:
            return query
        
        # 컨텍스트 결합
        context_text = "\n\n".join([f"문서 {i+1}:\n{doc}" for i, doc in enumerate(context)])
        
        # 프롬프트 생성
        prompt = f"""다음 문서를 참고하여 질문에 답변해주세요:

{context_text}

질문: {query}

답변:"""
        
        return prompt
    
    def _create_messages(self, query: str, context: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """
        메시지 생성
        
        Args:
            query: 질의 텍스트
            context: 컨텍스트 문서 목록
            
        Returns:
            메시지 목록
        """
        if not context:
            return [
                {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."},
                {"role": "user", "content": query}
            ]
        
        # 컨텍스트 결합
        context_text = "\n\n".join([f"문서 {i+1}:\n{doc}" for i, doc in enumerate(context)])
        
        # 메시지 생성
        messages = [
            {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다. 주어진 문서를 참고하여 질문에 답변해주세요."},
            {"role": "user", "content": f"다음 문서를 참고하여 질문에 답변해주세요:\n\n{context_text}\n\n질문: {query}"}
        ]
        
        return messages
    
    def generate(self, query: str, context: Optional[List[str]] = None, **kwargs) -> str:
        """
        응답 생성
        
        Args:
            query: 질의 텍스트
            context: 컨텍스트 문서 목록 (기본값: None)
            **kwargs: 추가 매개변수
            
        Returns:
            생성된 응답
        """
        # 매개변수 병합
        params = {**self.kwargs, **kwargs}
        
        # 온도 설정
        temperature = params.pop("temperature", self.temperature)
        
        # 최대 토큰 수 설정
        max_tokens = params.pop("max_tokens", self.max_tokens)
        
        try:
            # 메시지 생성
            messages = self._create_messages(query, context)
            
            # 응답 생성
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **params
            )
            
            # 응답 텍스트 추출
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {e}")
            return f"오류가 발생했습니다: {e}"
    
    def get_available_models(self) -> List[str]:
        """
        사용 가능한 OpenAI 질의 모델 목록 반환
        
        Returns:
            모델 이름 목록
        """
        # 모델 레지스트리에서 최신 정보 가져오기 시도
        self.model_registry.update_models_from_api(self.api_key)
        
        # 질의 모델 목록 반환
        return self.model_registry.get_all_query_models()
    
    def __str__(self) -> str:
        """모델 정보 문자열 반환"""
        return f"OpenAIQuery(model={self.model}, temperature={self.temperature}, max_tokens={self.max_tokens})" 