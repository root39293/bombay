import time
import os
from openai import OpenAI
from typing import Dict, List, Optional, Any
import json
import logging

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    OpenAI API 모델을 자동으로 관리하는 클래스입니다.
    사용 가능한 모델 목록을 가져오고, 카테고리별로 분류하며, 최신 모델을 반환합니다.
    """
    
    def __init__(self, api_key: str):
        """
        ModelRegistry 초기화
        
        Args:
            api_key: OpenAI API 키
        """
        self.client = OpenAI(api_key=api_key)
        self.models_cache: Dict[str, List[str]] = {}
        self.last_update: Optional[float] = None
        self.update_interval = 86400  # 1일(초 단위)
        self.cache_file = os.path.join(os.path.dirname(__file__), "models_cache.json")
        self._load_cache()
    
    def _load_cache(self) -> None:
        """캐시 파일에서 모델 정보를 로드합니다."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self.models_cache = cache_data.get('models', {})
                    self.last_update = cache_data.get('last_update')
                    logger.info(f"모델 캐시를 로드했습니다. 마지막 업데이트: {self.last_update}")
            except Exception as e:
                logger.warning(f"캐시 로드 중 오류 발생: {e}")
    
    def _save_cache(self) -> None:
        """모델 정보를 캐시 파일에 저장합니다."""
        try:
            cache_dir = os.path.dirname(self.cache_file)
            os.makedirs(cache_dir, exist_ok=True)
            
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'models': self.models_cache,
                    'last_update': self.last_update
                }, f)
            logger.info("모델 캐시를 저장했습니다.")
        except Exception as e:
            logger.warning(f"캐시 저장 중 오류 발생: {e}")

    def get_available_models(self, force_update: bool = False) -> Dict[str, List[str]]:
        """
        사용 가능한 모델 목록을 가져옵니다.
        
        Args:
            force_update: 강제로 모델 목록을 업데이트할지 여부
            
        Returns:
            카테고리별로 분류된 모델 목록
        """
        current_time = time.time()
        if (force_update or 
            not self.last_update or 
            not self.models_cache or
            (current_time - self.last_update > self.update_interval)):
            try:
                logger.info("OpenAI API에서 모델 목록을 업데이트합니다.")
                response = self.client.models.list()
                self.models_cache = self._categorize_models(response.data)
                self.last_update = current_time
                self._save_cache()
            except Exception as e:
                logger.error(f"모델 목록 업데이트 중 오류 발생: {e}")
                if not self.models_cache:  # 캐시가 비어있는 경우에만 예외 발생
                    # 기본 모델 목록 제공
                    self.models_cache = {
                        "embedding": ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
                        "chat": ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
                        "completion": ["gpt-3.5-turbo-instruct"],
                        "other": []
                    }
                    logger.info("기본 모델 목록을 사용합니다.")
        
        return self.models_cache

    def _categorize_models(self, models: List[Any]) -> Dict[str, List[str]]:
        """
        모델을 카테고리별로 분류합니다.
        
        Args:
            models: OpenAI API에서 반환한 모델 목록
            
        Returns:
            카테고리별로 분류된 모델 목록
        """
        categorized = {
            "embedding": [],
            "chat": [],
            "completion": [],
            "other": []
        }
        
        for model in models:
            model_id = model.id
            if "embedding" in model_id:
                categorized["embedding"].append(model_id)
            elif any(name in model_id for name in ["gpt-4", "gpt-3.5"]) and "instruct" not in model_id:
                categorized["chat"].append(model_id)
            elif "instruct" in model_id or "davinci" in model_id:
                categorized["completion"].append(model_id)
            else:
                categorized["other"].append(model_id)
                
        # 각 카테고리 내에서 모델 정렬
        for category in categorized:
            categorized[category].sort(reverse=True)
            
        return categorized

    def get_latest_model(self, category: str) -> Optional[str]:
        """
        특정 카테고리의 최신 모델을 반환합니다.
        
        Args:
            category: 모델 카테고리 (embedding, chat, completion, other)
            
        Returns:
            최신 모델 이름 또는 None
        """
        models = self.get_available_models()
        if category in models and models[category]:
            return models[category][0]
        return None
    
    def get_recommended_model(self, category: str) -> str:
        """
        특정 카테고리의 권장 모델을 반환합니다.
        최신 모델이 항상 최선의 선택이 아닐 수 있으므로, 권장 모델을 별도로 제공합니다.
        
        Args:
            category: 모델 카테고리 (embedding, chat, completion)
            
        Returns:
            권장 모델 이름
        """
        recommended = {
            "embedding": "text-embedding-3-small",  # 비용 효율적인 선택
            "chat": "gpt-3.5-turbo",  # 일반적인 용도에 적합
            "completion": "gpt-3.5-turbo-instruct"  # 완성 작업에 적합
        }
        
        models = self.get_available_models()
        if category in recommended:
            recommended_model = recommended[category]
            # 권장 모델이 사용 가능한지 확인
            if category in models and recommended_model in models[category]:
                return recommended_model
            # 사용 불가능한 경우 최신 모델 반환
            elif category in models and models[category]:
                return models[category][0]
        
        # 기본값 반환
        default_models = {
            "embedding": "text-embedding-ada-002",
            "chat": "gpt-3.5-turbo",
            "completion": "gpt-3.5-turbo-instruct"
        }
        return default_models.get(category, "gpt-3.5-turbo") 