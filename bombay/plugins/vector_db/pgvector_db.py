"""
pgvector 벡터 데이터베이스 플러그인 모듈
"""

import logging
import os
import uuid
import json
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np

# 벡터 데이터베이스 기본 클래스 임포트
from bombay.vector_db.vector_db import VectorDB

logger = logging.getLogger(__name__)

class PGVectorDB(VectorDB):
    """PostgreSQL pgvector 벡터 데이터베이스 클래스"""
    
    def __init__(self, 
                 dim: int = 1536, 
                 connection_string: Optional[str] = None,
                 table_name: str = "bombay_embeddings",
                 create_table: bool = True,
                 similarity: str = "cosine",
                 **kwargs):
        """
        PostgreSQL pgvector 벡터 데이터베이스 초기화
        
        Args:
            dim: 벡터 차원
            connection_string: PostgreSQL 연결 문자열
            table_name: 테이블 이름
            create_table: 테이블이 없을 경우 생성 여부
            similarity: 유사도 측정 방법 ('cosine', 'l2', 'inner')
            **kwargs: 추가 매개변수
        """
        super().__init__(dim=dim, **kwargs)
        
        # PostgreSQL 설정
        self.connection_string = connection_string or os.environ.get("PG_CONNECTION_STRING")
        if not self.connection_string:
            raise ValueError("PostgreSQL 연결 문자열이 필요합니다. connection_string 매개변수를 통해 전달하거나 PG_CONNECTION_STRING 환경 변수를 설정하세요.")
        
        self.table_name = table_name
        self.create_table = create_table
        
        # 유사도 매핑
        similarity_map = {
            "cosine": "cosine",
            "l2": "l2",
            "euclidean": "l2",
            "inner": "inner",
            "dotproduct": "inner",
            "dot": "inner"
        }
        self.similarity = similarity_map.get(similarity.lower(), "cosine")
        
        # PostgreSQL 초기화
        self._init_pgvector()
    
    def _init_pgvector(self) -> None:
        """PostgreSQL pgvector 초기화"""
        try:
            import psycopg2
            from psycopg2 import sql
            from psycopg2.extras import Json
            
            # 연결 테스트
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    # pgvector 확장 확인 및 설치
                    cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                    if cur.fetchone() is None:
                        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                        conn.commit()
                        logger.info("pgvector 확장을 설치했습니다.")
                    
                    # 테이블 확인 및 생성
                    cur.execute(
                        sql.SQL("SELECT 1 FROM information_schema.tables WHERE table_name = %s"),
                        (self.table_name,)
                    )
                    
                    if cur.fetchone() is None:
                        if self.create_table:
                            # 테이블 생성
                            cur.execute(
                                sql.SQL("""
                                CREATE TABLE IF NOT EXISTS {} (
                                    id TEXT PRIMARY KEY,
                                    embedding VECTOR({}),
                                    document TEXT,
                                    metadata JSONB
                                )
                                """).format(sql.Identifier(self.table_name), sql.Literal(self.dim))
                            )
                            
                            # 인덱스 생성
                            if self.similarity == "cosine":
                                cur.execute(
                                    sql.SQL("""
                                    CREATE INDEX IF NOT EXISTS {}_cosine_idx ON {} 
                                    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
                                    """).format(sql.Identifier(self.table_name), sql.Identifier(self.table_name))
                                )
                            elif self.similarity == "l2":
                                cur.execute(
                                    sql.SQL("""
                                    CREATE INDEX IF NOT EXISTS {}_l2_idx ON {} 
                                    USING ivfflat (embedding vector_l2_ops) WITH (lists = 100)
                                    """).format(sql.Identifier(self.table_name), sql.Identifier(self.table_name))
                                )
                            else:  # inner product
                                cur.execute(
                                    sql.SQL("""
                                    CREATE INDEX IF NOT EXISTS {}_inner_idx ON {} 
                                    USING ivfflat (embedding vector_ip_ops) WITH (lists = 100)
                                    """).format(sql.Identifier(self.table_name), sql.Identifier(self.table_name))
                                )
                            
                            conn.commit()
                            logger.info(f"테이블 '{self.table_name}'을 생성했습니다.")
                        else:
                            raise ValueError(f"테이블 '{self.table_name}'이 존재하지 않습니다. create_table=True로 설정하여 자동 생성하세요.")
            
            logger.info(f"PostgreSQL pgvector에 연결했습니다. 테이블: {self.table_name}")
        
        except ImportError:
            raise ImportError("psycopg2 패키지가 설치되지 않았습니다. 'pip install psycopg2-binary>=2.9.0'를 실행하여 설치하세요.")
        
        except Exception as e:
            logger.error(f"PostgreSQL pgvector 초기화 중 오류 발생: {e}")
            raise
    
    def add(self, 
            vectors: List[List[float]], 
            documents: List[str], 
            metadatas: Optional[List[Dict[str, Any]]] = None, 
            ids: Optional[List[str]] = None) -> List[str]:
        """
        벡터 추가
        
        Args:
            vectors: 벡터 목록
            documents: 문서 목록
            metadatas: 메타데이터 목록
            ids: ID 목록
            
        Returns:
            추가된 문서의 ID 목록
        """
        if not vectors or not documents:
            return []
        
        # 입력 검증
        if len(vectors) != len(documents):
            raise ValueError(f"벡터 수({len(vectors)})와 문서 수({len(documents)})가 일치하지 않습니다.")
        
        # 메타데이터 확인
        if metadatas is None:
            metadatas = [{} for _ in range(len(vectors))]
        elif len(metadatas) != len(vectors):
            raise ValueError(f"벡터 수({len(vectors)})와 메타데이터 수({len(metadatas)})가 일치하지 않습니다.")
        
        # ID 확인
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        elif len(ids) != len(vectors):
            raise ValueError(f"벡터 수({len(vectors)})와 ID 수({len(ids)})가 일치하지 않습니다.")
        
        try:
            import psycopg2
            from psycopg2 import sql
            from psycopg2.extras import Json
            
            # 데이터 삽입
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    # 배치 처리
                    for i in range(len(vectors)):
                        # 기존 ID가 있으면 업데이트, 없으면 삽입
                        cur.execute(
                            sql.SQL("""
                            INSERT INTO {} (id, embedding, document, metadata)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (id) DO UPDATE
                            SET embedding = EXCLUDED.embedding,
                                document = EXCLUDED.document,
                                metadata = EXCLUDED.metadata
                            """).format(sql.Identifier(self.table_name)),
                            (ids[i], vectors[i], documents[i], Json(metadatas[i]))
                        )
                
                conn.commit()
            
            return ids
        
        except Exception as e:
            logger.error(f"벡터 추가 중 오류 발생: {e}")
            return []
    
    def search(self, 
               query_vector: List[float], 
               k: int = 4, 
               filter_criteria: Optional[Dict[str, Any]] = None) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """
        벡터 검색
        
        Args:
            query_vector: 쿼리 벡터
            k: 검색할 문서 수
            filter_criteria: 필터링 기준
            
        Returns:
            (문서 목록, 거리 목록, 메타데이터 목록) 튜플
        """
        try:
            import psycopg2
            from psycopg2 import sql
            from psycopg2.extras import Json
            
            # 유사도 함수 선택
            if self.similarity == "cosine":
                similarity_func = "1 - (embedding <=> %s)"
            elif self.similarity == "l2":
                similarity_func = "1 / (1 + (embedding <-> %s))"
            else:  # inner product
                similarity_func = "embedding <#> %s"
            
            # 필터 쿼리 생성
            filter_query = ""
            filter_params = []
            
            if filter_criteria:
                filter_conditions = []
                
                for key, value in filter_criteria.items():
                    filter_conditions.append(f"metadata->>{len(filter_params) + 1} = %s")
                    filter_params.append(key)
                    filter_params.append(str(value))
                
                if filter_conditions:
                    filter_query = "WHERE " + " AND ".join(filter_conditions)
            
            # 검색 쿼리 실행
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    query = sql.SQL("""
                    SELECT document, {}, metadata
                    FROM {}
                    {}
                    ORDER BY {} DESC
                    LIMIT %s
                    """).format(
                        sql.SQL(similarity_func),
                        sql.Identifier(self.table_name),
                        sql.SQL(filter_query),
                        sql.SQL(similarity_func)
                    )
                    
                    params = [query_vector] + filter_params + [query_vector, k]
                    cur.execute(query, params)
                    
                    results = cur.fetchall()
            
            # 결과 추출
            documents = []
            distances = []
            metadatas = []
            
            for row in results:
                documents.append(row[0])
                distances.append(float(row[1]))
                metadatas.append(row[2])
            
            return documents, distances, metadatas
        
        except Exception as e:
            logger.error(f"벡터 검색 중 오류 발생: {e}")
            return [], [], []
    
    def delete(self, ids: List[str]) -> None:
        """
        벡터 삭제
        
        Args:
            ids: 삭제할 문서의 ID 목록
        """
        if not ids:
            return
        
        try:
            import psycopg2
            from psycopg2 import sql
            
            # 벡터 삭제
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    placeholders = ", ".join(["%s"] * len(ids))
                    cur.execute(
                        sql.SQL("DELETE FROM {} WHERE id IN ({})").format(
                            sql.Identifier(self.table_name),
                            sql.SQL(placeholders)
                        ),
                        ids
                    )
                
                conn.commit()
        
        except Exception as e:
            logger.error(f"벡터 삭제 중 오류 발생: {e}")
    
    def clear(self) -> None:
        """모든 벡터 삭제"""
        try:
            import psycopg2
            from psycopg2 import sql
            
            # 모든 벡터 삭제
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("TRUNCATE TABLE {}").format(sql.Identifier(self.table_name))
                    )
                
                conn.commit()
            
            logger.info(f"테이블 '{self.table_name}'의 모든 벡터를 삭제했습니다.")
        
        except Exception as e:
            logger.error(f"벡터 삭제 중 오류 발생: {e}")
    
    def count(self) -> int:
        """
        벡터 수 반환
        
        Returns:
            벡터 수
        """
        try:
            import psycopg2
            from psycopg2 import sql
            
            # 벡터 수 조회
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(self.table_name))
                    )
                    
                    result = cur.fetchone()
                    return result[0] if result else 0
        
        except Exception as e:
            logger.error(f"벡터 수 조회 중 오류 발생: {e}")
            return 0
    
    def save(self, path: str) -> None:
        """
        벡터 데이터베이스 저장 (지원하지 않음)
        
        Args:
            path: 저장 경로
        """
        logger.warning("PostgreSQL pgvector는 로컬 저장을 지원하지 않습니다. 데이터는 PostgreSQL 데이터베이스에 저장됩니다.")
    
    def load(self, path: str) -> None:
        """
        벡터 데이터베이스 로드 (지원하지 않음)
        
        Args:
            path: 로드 경로
        """
        logger.warning("PostgreSQL pgvector는 로컬 로드를 지원하지 않습니다. 데이터는 PostgreSQL 데이터베이스에서 로드됩니다.")
    
    def __str__(self) -> str:
        """문자열 표현"""
        return f"PGVectorDB(table={self.table_name}, similarity={self.similarity}, count={self.count()})" 