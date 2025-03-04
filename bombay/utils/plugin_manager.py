import os
import importlib
import inspect
from typing import Dict, Any, Type, Callable, List, Optional
import logging

logger = logging.getLogger(__name__)

# 플러그인 디렉토리 경로
PLUGIN_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plugins")

# 플러그인 캐시
_vector_db_plugins = {}
_embedding_model_plugins = {}
_query_model_plugins = {}
_document_loader_plugins = {}
_chunker_plugins = {}

def _ensure_plugin_dir(subdir: str = "") -> str:
    """
    플러그인 디렉토리가 존재하는지 확인하고, 없으면 생성합니다.
    
    Args:
        subdir: 서브 디렉토리 이름 (기본값: "")
        
    Returns:
        플러그인 디렉토리 경로
    """
    plugin_path = os.path.join(PLUGIN_DIR, subdir)
    os.makedirs(plugin_path, exist_ok=True)
    
    # __init__.py 파일이 없으면 생성
    init_file = os.path.join(plugin_path, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write("# 플러그인 패키지\n")
    
    return plugin_path

def _load_plugins_from_dir(dir_path: str, base_class: Type) -> Dict[str, Type]:
    """
    지정된 디렉토리에서 플러그인을 로드합니다.
    
    Args:
        dir_path: 플러그인 디렉토리 경로
        base_class: 플러그인의 기본 클래스
        
    Returns:
        플러그인 이름과 클래스의 딕셔너리
    """
    plugins = {}
    
    if not os.path.exists(dir_path):
        return plugins
    
    # 디렉토리 내의 모든 Python 파일 탐색
    for filename in os.listdir(dir_path):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]
            try:
                # 상대 경로로 모듈 임포트
                module_path = f"bombay.plugins.{os.path.basename(dir_path)}.{module_name}"
                module = importlib.import_module(module_path)
                
                # 모듈 내의 모든 클래스 검사
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (inspect.isclass(attr) and 
                        issubclass(attr, base_class) and 
                        attr is not base_class):
                        # 클래스 이름을 소문자로 변환하여 플러그인 이름으로 사용
                        plugin_name = attr_name.lower()
                        plugins[plugin_name] = attr
                        logger.info(f"플러그인 로드됨: {plugin_name} ({module_path}.{attr_name})")
            except Exception as e:
                logger.error(f"플러그인 로드 중 오류 발생: {module_name}, {e}")
    
    return plugins

def load_vector_db_plugins() -> Dict[str, Type]:
    """
    벡터 데이터베이스 플러그인을 로드합니다.
    
    Returns:
        벡터 데이터베이스 플러그인 딕셔너리
    """
    global _vector_db_plugins
    
    if not _vector_db_plugins:
        from ..pipeline.vector_db import VectorDB
        plugin_dir = _ensure_plugin_dir("vector_db")
        _vector_db_plugins = _load_plugins_from_dir(plugin_dir, VectorDB)
    
    return _vector_db_plugins

def get_vector_db_plugin(name: str) -> Optional[Type]:
    """
    이름으로 벡터 데이터베이스 플러그인을 가져옵니다.
    
    Args:
        name: 플러그인 이름
        
    Returns:
        벡터 데이터베이스 플러그인 클래스 또는 None
    """
    plugins = load_vector_db_plugins()
    return plugins.get(name.lower())

def load_embedding_model_plugins() -> Dict[str, Type]:
    """
    임베딩 모델 플러그인을 로드합니다.
    
    Returns:
        임베딩 모델 플러그인 딕셔너리
    """
    global _embedding_model_plugins
    
    if not _embedding_model_plugins:
        from ..pipeline.embedding_models import EmbeddingModel
        plugin_dir = _ensure_plugin_dir("embedding_models")
        _embedding_model_plugins = _load_plugins_from_dir(plugin_dir, EmbeddingModel)
    
    return _embedding_model_plugins

def get_embedding_model_plugins(api_key: str) -> Dict[str, Callable]:
    """
    임베딩 모델 플러그인 팩토리 함수를 가져옵니다.
    
    Args:
        api_key: API 키
        
    Returns:
        임베딩 모델 플러그인 팩토리 함수 딕셔너리
    """
    plugins = load_embedding_model_plugins()
    return {name: lambda plugin=plugin_class: plugin(api_key) 
            for name, plugin_class in plugins.items()}

def load_query_model_plugins() -> Dict[str, Type]:
    """
    질의 모델 플러그인을 로드합니다.
    
    Returns:
        질의 모델 플러그인 딕셔너리
    """
    global _query_model_plugins
    
    if not _query_model_plugins:
        from ..pipeline.query_models import QueryModel
        plugin_dir = _ensure_plugin_dir("query_models")
        _query_model_plugins = _load_plugins_from_dir(plugin_dir, QueryModel)
    
    return _query_model_plugins

def get_query_model_plugins(api_key: str) -> Dict[str, Callable]:
    """
    질의 모델 플러그인 팩토리 함수를 가져옵니다.
    
    Args:
        api_key: API 키
        
    Returns:
        질의 모델 플러그인 팩토리 함수 딕셔너리
    """
    plugins = load_query_model_plugins()
    return {name: lambda plugin=plugin_class: plugin(api_key) 
            for name, plugin_class in plugins.items()}

def load_document_loader_plugins() -> Dict[str, Type]:
    """
    문서 로더 플러그인을 로드합니다.
    
    Returns:
        문서 로더 플러그인 딕셔너리
    """
    global _document_loader_plugins
    
    if not _document_loader_plugins:
        from ..document_processing.loaders import DocumentLoader
        plugin_dir = _ensure_plugin_dir("document_loaders")
        _document_loader_plugins = _load_plugins_from_dir(plugin_dir, DocumentLoader)
    
    return _document_loader_plugins

def get_document_loader_plugin(name: str) -> Optional[Type]:
    """
    이름으로 문서 로더 플러그인을 가져옵니다.
    
    Args:
        name: 플러그인 이름
        
    Returns:
        문서 로더 플러그인 클래스 또는 None
    """
    plugins = load_document_loader_plugins()
    return plugins.get(name.lower())

def load_chunker_plugins() -> Dict[str, Type]:
    """
    청크 분할기 플러그인을 로드합니다.
    
    Returns:
        청크 분할기 플러그인 딕셔너리
    """
    global _chunker_plugins
    
    if not _chunker_plugins:
        from ..document_processing.chunkers import Chunker
        plugin_dir = _ensure_plugin_dir("chunkers")
        _chunker_plugins = _load_plugins_from_dir(plugin_dir, Chunker)
    
    return _chunker_plugins

def get_chunker_plugin(name: str) -> Optional[Type]:
    """
    이름으로 청크 분할기 플러그인을 가져옵니다.
    
    Args:
        name: 플러그인 이름
        
    Returns:
        청크 분할기 플러그인 클래스 또는 None
    """
    plugins = load_chunker_plugins()
    return plugins.get(name.lower()) 