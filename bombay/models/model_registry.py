import logging
from typing import Dict, List, Optional, Set, Tuple, Union
import os
import json
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelRegistry:
    """OpenAI 모델 레지스트리 클래스"""
    
    # 모델 캐시 파일 경로
    CACHE_DIR = Path.home() / ".bombay" / "cache"
    CACHE_FILE = CACHE_DIR / "model_cache.json"
    
    # 캐시 만료 시간 (24시간)
    CACHE_EXPIRY = 24 * 60 * 60
    
    def __init__(self):
        """모델 레지스트리 초기화"""
        # 임베딩 모델 정보
        self.embedding_models = {
            # 최신 임베딩 모델
            "text-embedding-3-large": {
                "dimensions": 3072,
                "max_tokens": 8191,
                "is_latest": True
            },
            "text-embedding-3-small": {
                "dimensions": 1536,
                "max_tokens": 8191,
                "is_latest": True
            },
            "text-embedding-ada-002": {
                "dimensions": 1536,
                "max_tokens": 8191,
                "is_latest": False
            }
        }
        
        # 질의 모델 정보
        self.query_models = {
            # GPT-4.5 모델
            "gpt-4.5-preview": {
                "alias": "gpt-4.5-preview-2025-02-27",
                "context_window": 128000,
                "max_output_tokens": 16384,
                "supports_vision": True,
                "supports_function_calling": True,
                "knowledge_cutoff": "2023-10",
                "category": "gpt-4.5"
            },
            "gpt-4.5-preview-2025-02-27": {
                "context_window": 128000,
                "max_output_tokens": 16384,
                "supports_vision": True,
                "supports_function_calling": True,
                "knowledge_cutoff": "2023-10",
                "category": "gpt-4.5"
            },
            
            # GPT-4o 모델
            "gpt-4o": {
                "alias": "gpt-4o-2024-08-06",
                "context_window": 128000,
                "max_output_tokens": 16384,
                "supports_vision": True,
                "supports_function_calling": True,
                "knowledge_cutoff": "2023-10",
                "category": "gpt-4o"
            },
            "gpt-4o-2024-11-20": {
                "context_window": 128000,
                "max_output_tokens": 16384,
                "supports_vision": True,
                "supports_function_calling": True,
                "knowledge_cutoff": "2023-10",
                "category": "gpt-4o"
            },
            "gpt-4o-2024-08-06": {
                "context_window": 128000,
                "max_output_tokens": 16384,
                "supports_vision": True,
                "supports_function_calling": True,
                "knowledge_cutoff": "2023-10",
                "category": "gpt-4o"
            },
            "gpt-4o-2024-05-13": {
                "context_window": 128000,
                "max_output_tokens": 4096,
                "supports_vision": True,
                "supports_function_calling": True,
                "knowledge_cutoff": "2023-10",
                "category": "gpt-4o"
            },
            
            # GPT-4o mini 모델
            "gpt-4o-mini": {
                "alias": "gpt-4o-mini-2024-07-18",
                "context_window": 128000,
                "max_output_tokens": 16384,
                "supports_vision": True,
                "supports_function_calling": True,
                "knowledge_cutoff": "2023-10",
                "category": "gpt-4o-mini"
            },
            "gpt-4o-mini-2024-07-18": {
                "context_window": 128000,
                "max_output_tokens": 16384,
                "supports_vision": True,
                "supports_function_calling": True,
                "knowledge_cutoff": "2023-10",
                "category": "gpt-4o-mini"
            },
            
            # o1 시리즈 모델
            "o1": {
                "alias": "o1-2024-12-17",
                "context_window": 200000,
                "max_output_tokens": 100000,
                "supports_vision": True,
                "supports_function_calling": True,
                "knowledge_cutoff": "2023-10",
                "category": "o1"
            },
            "o1-2024-12-17": {
                "context_window": 200000,
                "max_output_tokens": 100000,
                "supports_vision": True,
                "supports_function_calling": True,
                "knowledge_cutoff": "2023-10",
                "category": "o1"
            },
            "o1-mini": {
                "alias": "o1-mini-2024-09-12",
                "context_window": 128000,
                "max_output_tokens": 65536,
                "supports_vision": False,
                "supports_function_calling": True,
                "knowledge_cutoff": "2023-10",
                "category": "o1-mini"
            },
            "o1-mini-2024-09-12": {
                "context_window": 128000,
                "max_output_tokens": 65536,
                "supports_vision": False,
                "supports_function_calling": True,
                "knowledge_cutoff": "2023-10",
                "category": "o1-mini"
            },
            
            # o3-mini 모델
            "o3-mini": {
                "alias": "o3-mini-2025-01-31",
                "context_window": 200000,
                "max_output_tokens": 100000,
                "supports_vision": False,
                "supports_function_calling": True,
                "knowledge_cutoff": "2023-10",
                "category": "o3-mini"
            },
            "o3-mini-2025-01-31": {
                "context_window": 200000,
                "max_output_tokens": 100000,
                "supports_vision": False,
                "supports_function_calling": True,
                "knowledge_cutoff": "2023-10",
                "category": "o3-mini"
            },
            
            # GPT-4 Turbo 모델
            "gpt-4-turbo": {
                "alias": "gpt-4-turbo-2024-04-09",
                "context_window": 128000,
                "max_output_tokens": 4096,
                "supports_vision": True,
                "supports_function_calling": True,
                "knowledge_cutoff": "2023-12",
                "category": "gpt-4-turbo"
            },
            "gpt-4-turbo-2024-04-09": {
                "context_window": 128000,
                "max_output_tokens": 4096,
                "supports_vision": True,
                "supports_function_calling": True,
                "knowledge_cutoff": "2023-12",
                "category": "gpt-4-turbo"
            },
            
            # GPT-4 모델
            "gpt-4": {
                "alias": "gpt-4-0613",
                "context_window": 8192,
                "max_output_tokens": 8192,
                "supports_vision": False,
                "supports_function_calling": True,
                "knowledge_cutoff": "2021-09",
                "category": "gpt-4"
            },
            "gpt-4-0613": {
                "context_window": 8192,
                "max_output_tokens": 8192,
                "supports_vision": False,
                "supports_function_calling": True,
                "knowledge_cutoff": "2021-09",
                "category": "gpt-4"
            },
            
            # GPT-3.5 Turbo 모델
            "gpt-3.5-turbo": {
                "alias": "gpt-3.5-turbo-0125",
                "context_window": 16385,
                "max_output_tokens": 4096,
                "supports_vision": False,
                "supports_function_calling": True,
                "knowledge_cutoff": "2021-09",
                "category": "gpt-3.5-turbo"
            },
            "gpt-3.5-turbo-0125": {
                "context_window": 16385,
                "max_output_tokens": 4096,
                "supports_vision": False,
                "supports_function_calling": True,
                "knowledge_cutoff": "2021-09",
                "category": "gpt-3.5-turbo"
            }
        }
        
        # 모델 카테고리 그룹
        self.model_categories = {
            "gpt-3": ["gpt-3.5-turbo"],
            "gpt-4": ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4.5-preview"],
            "reasoning": ["o1", "o1-mini", "o3-mini"],
            "openai": ["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"]
        }
        
        # 캐시 디렉토리 생성
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # 캐시에서 모델 정보 로드 시도
        self._load_from_cache()
    
    def _load_from_cache(self) -> bool:
        """캐시에서 모델 정보 로드"""
        try:
            if not self.CACHE_FILE.exists():
                return False
            
            # 캐시 파일 읽기
            with open(self.CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
            
            # 캐시 만료 확인
            if time.time() - cache_data.get('timestamp', 0) > self.CACHE_EXPIRY:
                logger.info("모델 캐시가 만료되었습니다.")
                return False
            
            # 캐시에서 모델 정보 업데이트
            if 'embedding_models' in cache_data:
                self.embedding_models.update(cache_data['embedding_models'])
            
            if 'query_models' in cache_data:
                self.query_models.update(cache_data['query_models'])
            
            logger.info("캐시에서 모델 정보를 로드했습니다.")
            return True
        
        except Exception as e:
            logger.warning(f"캐시에서 모델 정보를 로드하는 중 오류 발생: {e}")
            return False
    
    def _save_to_cache(self) -> None:
        """모델 정보를 캐시에 저장"""
        try:
            cache_data = {
                'timestamp': time.time(),
                'embedding_models': self.embedding_models,
                'query_models': self.query_models
            }
            
            with open(self.CACHE_FILE, 'w') as f:
                json.dump(cache_data, f)
            
            logger.info("모델 정보를 캐시에 저장했습니다.")
        
        except Exception as e:
            logger.warning(f"모델 정보를 캐시에 저장하는 중 오류 발생: {e}")
    
    def update_models_from_api(self, api_key: Optional[str] = None) -> bool:
        """OpenAI API에서 최신 모델 정보 가져오기"""
        try:
            import openai
            
            # API 키 설정
            if api_key:
                openai.api_key = api_key
            elif os.environ.get("OPENAI_API_KEY"):
                openai.api_key = os.environ.get("OPENAI_API_KEY")
            else:
                logger.warning("OpenAI API 키가 설정되지 않았습니다.")
                return False
            
            # 모델 목록 가져오기
            client = openai.OpenAI()
            models = client.models.list()
            
            # 모델 정보 업데이트
            for model in models.data:
                model_id = model.id
                
                # 임베딩 모델 업데이트
                if "embedding" in model_id:
                    # 기존 모델이 아니면 기본 정보로 추가
                    if model_id not in self.embedding_models:
                        self.embedding_models[model_id] = {
                            "dimensions": 1536,  # 기본값
                            "max_tokens": 8191,  # 기본값
                            "is_latest": "3" in model_id  # 버전 3이면 최신으로 간주
                        }
                
                # 질의 모델 업데이트
                elif any(prefix in model_id for prefix in ["gpt-", "o1", "o3"]):
                    # 기존 모델이 아니면 기본 정보로 추가
                    if model_id not in self.query_models:
                        category = "gpt-3.5-turbo"
                        if "gpt-4" in model_id:
                            category = "gpt-4"
                        elif "gpt-4o" in model_id:
                            category = "gpt-4o"
                        elif "gpt-4.5" in model_id:
                            category = "gpt-4.5"
                        elif "o1" in model_id:
                            category = "o1"
                        elif "o3" in model_id:
                            category = "o3-mini"
                        
                        self.query_models[model_id] = {
                            "context_window": 8192,  # 기본값
                            "max_output_tokens": 4096,  # 기본값
                            "supports_vision": "vision" in model_id or "4o" in model_id,
                            "supports_function_calling": True,  # 대부분 지원
                            "knowledge_cutoff": "2021-09",  # 기본값
                            "category": category
                        }
            
            # 캐시에 저장
            self._save_to_cache()
            
            logger.info("OpenAI API에서 모델 정보를 업데이트했습니다.")
            return True
        
        except Exception as e:
            logger.warning(f"OpenAI API에서 모델 정보를 업데이트하는 중 오류 발생: {e}")
            return False
    
    def get_latest_embedding_model(self) -> str:
        """최신 임베딩 모델 이름 반환"""
        for model_id, info in self.embedding_models.items():
            if info.get("is_latest", False) and "large" in model_id:
                return model_id
        
        # 기본값 반환
        return "text-embedding-3-large"
    
    def get_latest_query_model(self, category: str = "gpt-3") -> str:
        """지정된 카테고리의 최신 질의 모델 이름 반환"""
        if category not in self.model_categories:
            logger.warning(f"알 수 없는 모델 카테고리: {category}, 기본값 사용")
            category = "gpt-3"
        
        # 카테고리에 해당하는 모델 그룹 가져오기
        model_group = self.model_categories[category]
        
        # 각 그룹에서 가장 최신 모델 찾기
        latest_models = {}
        for group in model_group:
            latest_model = None
            latest_date = ""
            
            for model_id, info in self.query_models.items():
                if info.get("category") == group and "alias" not in info:
                    # 날짜 형식 모델(예: gpt-4o-2024-08-06)에서 날짜 추출
                    date_parts = model_id.split("-")
                    date_str = ""
                    for part in date_parts:
                        if part.isdigit() and len(part) == 4:  # 연도로 추정
                            idx = model_id.find(part)
                            if idx > 0 and len(model_id) > idx + 10:
                                date_str = model_id[idx:idx+10]
                                break
                    
                    if date_str and (not latest_date or date_str > latest_date):
                        latest_date = date_str
                        latest_model = model_id
            
            # 날짜 기반 모델을 찾지 못했으면 별칭 사용
            if not latest_model:
                for model_id, info in self.query_models.items():
                    if info.get("category") == group and "alias" not in model_id:
                        latest_model = model_id
                        break
            
            if latest_model:
                latest_models[group] = latest_model
        
        # 카테고리에 맞는 모델 반환
        if category == "gpt-3" and "gpt-3.5-turbo" in latest_models:
            return latest_models["gpt-3.5-turbo"]
        elif category == "gpt-4":
            # gpt-4o가 있으면 우선 반환
            if "gpt-4o" in latest_models:
                return latest_models["gpt-4o"]
            # 없으면 gpt-4-turbo 반환
            elif "gpt-4-turbo" in latest_models:
                return latest_models["gpt-4-turbo"]
            # 둘 다 없으면 gpt-4 반환
            elif "gpt-4" in latest_models:
                return latest_models["gpt-4"]
            # gpt-4.5-preview 반환
            elif "gpt-4.5-preview" in latest_models:
                return latest_models["gpt-4.5-preview"]
        elif category == "reasoning":
            # o3-mini가 있으면 우선 반환
            if "o3-mini" in latest_models:
                return latest_models["o3-mini"]
            # 없으면 o1 반환
            elif "o1" in latest_models:
                return latest_models["o1"]
            # 둘 다 없으면 o1-mini 반환
            elif "o1-mini" in latest_models:
                return latest_models["o1-mini"]
        
        # 기본값 반환
        if category == "gpt-3":
            return "gpt-3.5-turbo"
        elif category == "gpt-4":
            return "gpt-4o"
        elif category == "reasoning":
            return "o3-mini"
        else:
            return "gpt-3.5-turbo"
    
    def get_embedding_dimension(self, model_name: str) -> int:
        """임베딩 모델의 차원 반환"""
        if model_name in self.embedding_models:
            return self.embedding_models[model_name].get("dimensions", 1536)
        
        # 기본값 반환
        logger.warning(f"알 수 없는 임베딩 모델: {model_name}, 기본 차원 사용")
        return 1536
    
    def get_context_window(self, model_name: str) -> int:
        """질의 모델의 컨텍스트 윈도우 크기 반환"""
        if model_name in self.query_models:
            return self.query_models[model_name].get("context_window", 8192)
        
        # 기본값 반환
        logger.warning(f"알 수 없는 질의 모델: {model_name}, 기본 컨텍스트 윈도우 사용")
        return 8192
    
    def get_max_output_tokens(self, model_name: str) -> int:
        """질의 모델의 최대 출력 토큰 수 반환"""
        if model_name in self.query_models:
            return self.query_models[model_name].get("max_output_tokens", 4096)
        
        # 기본값 반환
        logger.warning(f"알 수 없는 질의 모델: {model_name}, 기본 최대 출력 토큰 수 사용")
        return 4096
    
    def supports_vision(self, model_name: str) -> bool:
        """질의 모델의 비전 지원 여부 반환"""
        if model_name in self.query_models:
            return self.query_models[model_name].get("supports_vision", False)
        
        # 기본값 반환
        return False
    
    def supports_function_calling(self, model_name: str) -> bool:
        """질의 모델의 함수 호출 지원 여부 반환"""
        if model_name in self.query_models:
            return self.query_models[model_name].get("supports_function_calling", False)
        
        # 기본값 반환
        return False
    
    def get_all_embedding_models(self) -> List[str]:
        """모든 임베딩 모델 이름 목록 반환"""
        return list(self.embedding_models.keys())
    
    def get_all_query_models(self) -> List[str]:
        """모든 질의 모델 이름 목록 반환"""
        return list(self.query_models.keys())
    
    def get_model_info(self, model_name: str) -> Dict:
        """모델 정보 반환"""
        if model_name in self.embedding_models:
            return self.embedding_models[model_name]
        elif model_name in self.query_models:
            return self.query_models[model_name]
        else:
            logger.warning(f"알 수 없는 모델: {model_name}")
            return {}
    
    def resolve_model_alias(self, model_name: str) -> str:
        """모델 별칭 해결 (별칭이면 실제 모델 이름 반환)"""
        if model_name in self.query_models and "alias" in self.query_models[model_name]:
            return self.query_models[model_name]["alias"]
        return model_name 