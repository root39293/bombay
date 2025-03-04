# document_processing_example.py
from bombay.pipeline import create_pipeline
from bombay.document_processing import DocumentProcessor, ParagraphChunker, SemanticChunker
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 파이프라인 생성
pipeline = create_pipeline(
    embedding_model_name='openai',
    query_model_name='gpt-3',
    vector_db='chromadb',
    api_key=api_key,
    use_persistent_storage=True,
    collection_name='document_example'
)

print("1. 다양한 청크 분할 전략 비교")
print("-" * 50)

# 샘플 텍스트
sample_text = """
인공지능(AI)은 인간의 학습, 추론, 지각, 문제 해결 능력 등을 컴퓨터 프로그램으로 구현한 기술입니다.
머신러닝은 컴퓨터가 데이터로부터 학습하여 패턴을 인식하고 예측하는 AI의 한 분야입니다.

딥러닝은 인공 신경망을 기반으로 하는 머신러닝의 한 종류로, 복잡한 패턴을 인식하는 데 뛰어납니다.
자연어 처리(NLP)는 컴퓨터가 인간의 언어를 이해하고 처리하는 AI 기술입니다.
컴퓨터 비전은 컴퓨터가 이미지나 비디오를 이해하고 해석하는 AI 기술입니다.

강화학습은 에이전트가 환경과 상호작용하며 보상을 최대화하는 방향으로 학습하는 머신러닝의 한 종류입니다.
전이학습은 하나의 작업에서 학습한 지식을 다른 관련 작업에 적용하는 기법입니다.
"""

# 다양한 청크 분할 전략 비교
chunkers = {
    "단순 분할": pipeline.document_processor.chunkers["simple"],
    "단락 분할": pipeline.document_processor.chunkers["paragraph"],
    "문장 분할": pipeline.document_processor.chunkers["sentence"]
}

for name, chunker in chunkers.items():
    chunks = chunker.split(sample_text, chunk_size=200, chunk_overlap=50)
    print(f"\n{name} 결과 (총 {len(chunks)}개 청크):")
    for i, chunk in enumerate(chunks):
        print(f"  청크 {i+1}: {chunk.content[:50]}... (길이: {len(chunk.content)}자)")

# 의미 기반 분할 (임베딩 모델 필요)
semantic_chunker = SemanticChunker(pipeline.embedding_model)
semantic_chunks = semantic_chunker.split(sample_text, chunk_size=200, chunk_overlap=50)
print(f"\n의미 기반 분할 결과 (총 {len(semantic_chunks)}개 청크):")
for i, chunk in enumerate(semantic_chunks):
    print(f"  청크 {i+1}: {chunk.content[:50]}... (길이: {len(chunk.content)}자)")
    print(f"    클러스터 ID: {chunk.metadata.get('cluster_id')}")

print("\n2. 파일 처리 예제")
print("-" * 50)

# 파일 처리 예제 (파일이 존재한다고 가정)
# 지원되는 파일 형식: txt, md, pdf, docx, html, csv
file_path = "example/sample.txt"

# 파일이 존재하는 경우에만 실행
if os.path.exists(file_path):
    # 파일 처리 및 청크 추출
    chunks = pipeline.document_processor.process_file(file_path)
    
    print(f"\n파일 '{file_path}'에서 추출된 청크 (총 {len(chunks)}개):")
    for i, chunk in enumerate(chunks[:3]):  # 처음 3개만 출력
        print(f"  청크 {i+1}: {chunk.content[:50]}...")
        print(f"    메타데이터: {chunk.metadata}")
    
    if len(chunks) > 3:
        print(f"  ... 외 {len(chunks) - 3}개 청크")
    
    # 파이프라인에 청크 추가
    pipeline.add_chunks(chunks)
    
    # 질의 응답
    query = "이 문서의 주요 내용은 무엇인가요?"
    result = pipeline.search_and_answer(query, k=2)
    
    print(f"\n질문: {result['query']}")
    print(f"답변: {result['answer']}")
else:
    print(f"파일 '{file_path}'가 존재하지 않습니다. 예제 파일을 생성하거나 경로를 수정하세요.") 