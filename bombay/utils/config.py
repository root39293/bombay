# bombay/utils/config.py
import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        설정 딕셔너리
    """
    if not os.path.exists(config_path):
        logger.warning(f"설정 파일 '{config_path}'이 존재하지 않습니다.")
        return {}
    
    try:
        # 파일 확장자 확인
        _, ext = os.path.splitext(config_path.lower())
        
        if ext == '.json':
            # JSON 파일 로드
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        
        elif ext in ['.yaml', '.yml']:
            # YAML 파일 로드
            try:
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            except ImportError:
                logger.error("YAML 파일을 로드하려면 'pip install pyyaml>=6.0.0'를 실행하여 패키지를 설치하세요.")
                return {}
        
        else:
            logger.error(f"지원하지 않는 설정 파일 형식: {ext}")
            return {}
        
        return config
    
    except Exception as e:
        logger.error(f"설정 파일 로드 중 오류 발생: {e}")
        return {}

def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    설정 파일 저장
    
    Args:
        config: 설정 딕셔너리
        config_path: 설정 파일 경로
        
    Returns:
        성공 여부
    """
    try:
        # 디렉토리 생성
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # 파일 확장자 확인
        _, ext = os.path.splitext(config_path.lower())
        
        if ext == '.json':
            # JSON 파일 저장
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        
        elif ext in ['.yaml', '.yml']:
            # YAML 파일 저장
            try:
                import yaml
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            except ImportError:
                logger.error("YAML 파일을 저장하려면 'pip install pyyaml>=6.0.0'를 실행하여 패키지를 설치하세요.")
                return False
        
        else:
            logger.error(f"지원하지 않는 설정 파일 형식: {ext}")
            return False
        
        logger.info(f"설정을 '{config_path}'에 저장했습니다.")
        return True
    
    except Exception as e:
        logger.error(f"설정 파일 저장 중 오류 발생: {e}")
        return False

def get_config_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    설정 값 가져오기
    
    Args:
        config: 설정 딕셔너리
        key: 키 (점으로 구분된 경로 지원)
        default: 기본값
        
    Returns:
        설정 값 또는 기본값
    """
    if not config:
        return default
    
    # 점으로 구분된 키 처리
    keys = key.split('.')
    value = config
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    
    return value

def update_config_value(config: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
    """
    설정 값 업데이트
    
    Args:
        config: 설정 딕셔너리
        key: 키 (점으로 구분된 경로 지원)
        value: 새 값
        
    Returns:
        업데이트된 설정 딕셔너리
    """
    if not config:
        config = {}
    
    # 점으로 구분된 키 처리
    keys = key.split('.')
    current = config
    
    # 마지막 키를 제외한 모든 키에 대해 딕셔너리 생성
    for i, k in enumerate(keys[:-1]):
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]
    
    # 마지막 키에 값 설정
    current[keys[-1]] = value
    
    return config