# bombay/pipeline/query_models.py
from abc import ABC, abstractmethod
from openai import OpenAI
from typing import List, Dict, Any, Optional, Union
from ..utils.model_registry import ModelRegistry

class QueryModel(ABC):
    @abstractmethod
    def generate(self, query: str, relevant_docs: List[str]) -> str:
        """
        쿼리와 관련 문서를 사용하여 답변을 생성하는 메소드
        
        Args:
            query: 사용자 쿼리
            relevant_docs: 관련 문서 리스트
            
        Returns:
            생성된 답변
        """
        pass


# GPT 기반 질의 모델 어댑터
class OpenAIQuery(QueryModel):
    def __init__(self, api_key: str, model: Optional[str] = None, temperature: float = 0.7):
        """
        GPT 기반 질의 모델 초기화
        
        Args:
            api_key: OpenAI API 키
            model: 사용할 GPT 모델 (기본값: None, 자동 선택)
            temperature: 생성 다양성 조절 파라미터 (0.0 ~ 1.0)
        """
        self.client = OpenAI(api_key=api_key)
        self.model_registry = ModelRegistry(api_key)
        
        # 모델이 지정되지 않은 경우 권장 모델 사용
        if model is None:
            self.model = self.model_registry.get_recommended_model("chat")
        else:
            self.model = model
            
        self.temperature = temperature
        self.default_prompt_template = """
        다음 관련 문서를 참고하여 질문에 답변해주세요.
        
        관련 문서:
        {relevant_docs}
        
        질문: {query}
        
        답변:
        """

    def generate(self, query: str, relevant_docs: List[str], prompt_template: Optional[str] = None) -> str:
        """
        쿼리와 관련 문서를 사용하여 GPT로 답변을 생성하는 메소드
        
        Args:
            query: 사용자 쿼리
            relevant_docs: 관련 문서 리스트
            prompt_template: 사용자 정의 프롬프트 템플릿 (기본값: None)
            
        Returns:
            생성된 답변
        """
        # 관련 문서를 하나의 문자열로 결합
        relevant_docs_str = '\n'.join([f"- {doc}" for doc in relevant_docs])
        
        # 프롬프트 템플릿 선택
        template = prompt_template if prompt_template else self.default_prompt_template
        
        # 템플릿에 값 채우기
        formatted_prompt = template.format(
            relevant_docs=relevant_docs_str,
            query=query
        )
        
        # 채팅 완성 요청
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "당신은 질문에 정확하게 답변하는 도우미입니다. 제공된 관련 문서의 정보만 사용하여 답변하세요."},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=self.temperature
        )
        
        return response.choices[0].message.content
    
    def get_available_models(self) -> List[str]:
        """
        사용 가능한 채팅 모델 목록을 반환하는 메소드
        
        Returns:
            사용 가능한 채팅 모델 목록
        """
        return self.model_registry.get_available_models()["chat"]
    
    def update_model(self, model: str) -> None:
        """
        사용할 채팅 모델을 업데이트하는 메소드
        
        Args:
            model: 새로운 채팅 모델 이름
        """
        self.model = model
    
    def set_prompt_template(self, template: str) -> None:
        """
        프롬프트 템플릿을 설정하는 메소드
        
        Args:
            template: 새로운 프롬프트 템플릿
        """
        self.default_prompt_template = template