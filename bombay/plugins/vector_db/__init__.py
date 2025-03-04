"""
벡터 데이터베이스 플러그인 패키지

이 패키지는 추가 벡터 데이터베이스 구현을 제공합니다.
"""

# 플러그인 목록
__all__ = []

# Pinecone 플러그인 (선택적)
try:
    from bombay.plugins.vector_db.pinecone_db import PineconeDB
    __all__.append('PineconeDB')
except ImportError:
    pass

# pgvector 플러그인 (선택적)
try:
    from bombay.plugins.vector_db.pgvector_db import PGVectorDB
    __all__.append('PGVectorDB')
except ImportError:
    pass 