# model_info_example.py
from bombay.models.model_registry import ModelRegistry
from bombay.models.embedding_model import OpenAIEmbedding
from bombay.models.query_model import OpenAIQuery
from bombay.pipeline import create_pipeline
from dotenv import load_dotenv
import os
from rich.console import Console
from rich.table import Table
from rich import print as rprint

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("OPENAI_API_KEY 환경 변수를 설정해야 합니다.")
    exit(1)

# 콘솔 초기화
console = Console()

print("OpenAI 모델 정보 예제")
print("-" * 50)

# 모델 레지스트리 초기화
model_registry = ModelRegistry()

# API에서 최신 모델 정보 가져오기
print("\nOpenAI API에서 최신 모델 정보를 가져오는 중...")
success = model_registry.update_models_from_api(api_key)

if success:
    print("✅ 모델 정보를 성공적으로 가져왔습니다.")
else:
    print("⚠️ 모델 정보를 가져오지 못했습니다. 캐시된 정보를 사용합니다.")

# 임베딩 모델 정보 출력
print("\n1. 임베딩 모델 정보")
print("-" * 40)

embedding_models = model_registry.get_all_embedding_models()
latest_embedding_model = model_registry.get_latest_embedding_model()

# 테이블 생성
table = Table(title="OpenAI 임베딩 모델")
table.add_column("모델 이름", style="cyan")
table.add_column("차원", style="green")
table.add_column("최대 토큰", style="yellow")
table.add_column("최신 모델", style="magenta")

for model in embedding_models:
    info = model_registry.get_model_info(model)
    is_latest = "✓" if info.get("is_latest", False) else ""
    latest_mark = "✓" if model == latest_embedding_model else ""
    
    table.add_row(
        model,
        str(info.get("dimensions", "-")),
        str(info.get("max_tokens", "-")),
        is_latest + latest_mark
    )

console.print(table)

# 질의 모델 정보 출력
print("\n2. 질의 모델 정보")
print("-" * 40)

# 모델 카테고리별 최신 모델 가져오기
latest_gpt3 = model_registry.get_latest_query_model("gpt-3")
latest_gpt4 = model_registry.get_latest_query_model("gpt-4")
latest_reasoning = model_registry.get_latest_query_model("reasoning")

print(f"최신 GPT-3 모델: {latest_gpt3}")
print(f"최신 GPT-4 모델: {latest_gpt4}")
print(f"최신 추론 모델: {latest_reasoning}")

# 주요 모델 정보 테이블 생성
table = Table(title="주요 OpenAI 질의 모델")
table.add_column("모델 이름", style="cyan")
table.add_column("컨텍스트 윈도우", style="green")
table.add_column("최대 출력 토큰", style="yellow")
table.add_column("비전 지원", style="magenta")
table.add_column("함수 호출 지원", style="blue")
table.add_column("지식 컷오프", style="red")

# 주요 모델 목록
main_models = [
    "gpt-4.5-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "o1",
    "o1-mini",
    "o3-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo"
]

for model in main_models:
    info = model_registry.get_model_info(model)
    if not info:
        continue
    
    alias = info.get("alias", "")
    alias_info = f" → {alias}" if alias else ""
    
    table.add_row(
        f"{model}{alias_info}",
        str(info.get("context_window", "-")),
        str(info.get("max_output_tokens", "-")),
        "✓" if info.get("supports_vision", False) else "✗",
        "✓" if info.get("supports_function_calling", False) else "✗",
        info.get("knowledge_cutoff", "-")
    )

console.print(table)

# 임베딩 모델 사용 예제
print("\n3. 임베딩 모델 사용 예제")
print("-" * 40)

# 다양한 임베딩 모델 생성
embedding_models = {
    "기본(최신)": OpenAIEmbedding(api_key=api_key),
    "text-embedding-3-large": OpenAIEmbedding(model="text-embedding-3-large", api_key=api_key),
    "text-embedding-3-small": OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key),
    "text-embedding-ada-002": OpenAIEmbedding(model="text-embedding-ada-002", api_key=api_key)
}

# 테스트 텍스트
test_text = "인공지능은 인간의 학습, 추론, 지각, 문제 해결 능력 등을 컴퓨터 프로그램으로 구현한 기술입니다."

# 각 모델의 임베딩 생성 및 정보 출력
for name, model in embedding_models.items():
    embedding = model.get_embedding(test_text)
    print(f"{name} 모델:")
    print(f"  - 모델 이름: {model.model}")
    print(f"  - 임베딩 차원: {model.get_dimension()}")
    print(f"  - 임베딩 벡터 (처음 5개 요소): {embedding[:5]}")
    print()

# 질의 모델 사용 예제
print("\n4. 질의 모델 사용 예제")
print("-" * 40)

# 다양한 질의 모델 생성
query_models = {
    "GPT-3 (최신)": OpenAIQuery(model="gpt-3", api_key=api_key),
    "GPT-4 (최신)": OpenAIQuery(model="gpt-4", api_key=api_key),
    "추론 모델 (최신)": OpenAIQuery(model="reasoning", api_key=api_key)
}

# 테스트 질의
test_query = "인공지능의 주요 분야 3가지를 간략히 설명해주세요."

# 각 모델의 응답 생성 및 정보 출력
for name, model in query_models.items():
    print(f"{name} 모델:")
    print(f"  - 모델 이름: {model.model}")
    print(f"  - 온도: {model.temperature}")
    print(f"  - 최대 토큰 수: {model.max_tokens}")
    
    try:
        # 응답 생성 (시간이 오래 걸릴 수 있음)
        print("  - 응답 생성 중...")
        response = model.generate(test_query, max_tokens=100)
        print(f"  - 응답: {response[:200]}...")
    except Exception as e:
        print(f"  - 오류 발생: {e}")
    
    print()

# 파이프라인 생성 예제
print("\n5. 파이프라인 생성 예제")
print("-" * 40)

# 다양한 모델 조합으로 파이프라인 생성
pipelines = {
    "기본 파이프라인": create_pipeline(api_key=api_key),
    "GPT-4o 파이프라인": create_pipeline(
        embedding_model_name="text-embedding-3-large",
        query_model_name="gpt-4o",
        vector_db="chromadb",
        api_key=api_key
    ),
    "추론 모델 파이프라인": create_pipeline(
        embedding_model_name="text-embedding-3-small",
        query_model_name="o3-mini",
        vector_db="hnswlib",
        api_key=api_key
    )
}

# 각 파이프라인 정보 출력
for name, pipeline in pipelines.items():
    print(f"{name}:")
    print(f"  - 임베딩 모델: {pipeline.embedding_model}")
    print(f"  - 질의 모델: {pipeline.query_model}")
    print(f"  - 벡터 데이터베이스: {pipeline.vector_db}")
    print()

print("\n모델 정보 예제 완료!")
print("이 예제를 통해 Bombay에서 지원하는 다양한 OpenAI 모델을 확인하고 사용하는 방법을 알아보았습니다.") 