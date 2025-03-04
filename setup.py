#setup.py
from setuptools import setup, find_packages

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bombay",
    version="1.0.0",
    author="Bombay Team",
    author_email="info@bombay.ai",
    description="RAG(Retrieval-Augmented Generation) 파이프라인을 쉽게 구축할 수 있는 Python 라이브러리",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bombay-ai/bombay",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=[
        "openai>=1.0.0",
        "numpy>=1.20.0",
        "pydantic>=2.10.0",
        "pydantic-core>=2.27.0",
        "httpx>=0.28.0",
        "hnswlib>=0.7.0",
        "chromadb>=0.4.0",
        "python-dotenv>=1.0.0",
        "pyfiglet>=0.8.0",
        "rich>=13.0.0",
        "scikit-learn>=1.0.0",
        "beautifulsoup4>=4.10.0",
        "markdown>=3.4.0",
        "PyPDF2>=3.0.0",
        "python-docx>=0.8.11",
        "requests>=2.28.0",
        "pyyaml>=6.0.0",
        "bs4>=0.0.1",
        "tqdm>=4.67.0",
    ],
    extras_require={
        "pinecone": ["pinecone-client>=2.2.0"],
        "pgvector": ["psycopg2-binary>=2.9.0"],
        "pdf": ["pymupdf>=1.20.0"],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
        "all": [
            "pinecone-client>=2.2.0",
            "psycopg2-binary>=2.9.0",
            "pymupdf>=1.20.0",
        ],
    },
)