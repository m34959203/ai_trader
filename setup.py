from setuptools import setup, find_packages

setup(
    name="ai_trader",                          # имя пакета
    version="0.1.0",                           # версия проекта
    description="AI-Trader: автономный трейдинг с анализом рынка и риск-менеджментом",
    author="Дмитрий",                          # можно заменить на своё имя / компанию
    author_email="you@example.com",            # контактный email
    url="https://github.com/yourrepo/ai_trader",  # если есть репозиторий
    package_dir={"": "src"},                   # исходники находятся в src/
    packages=find_packages(where="src"),       # ищем пакеты только внутри src/
    python_requires=">=3.11",                  # минимальная версия Python
    install_requires=[
        "fastapi>=0.111",
        "uvicorn>=0.30",
        "sqlalchemy>=2.0",
        "alembic>=1.13",
        "httpx>=0.28",
        "pandas>=2.2",
        "numpy>=1.26",
        "pytest>=8.2",
        "pytest-asyncio>=0.23",
        "asgi-lifespan>=2.1",
        "ccxt>=4.3",
        "loguru>=0.7",
        "python-dotenv>=1.0",
        "feedparser>=6.0",
        "prometheus-client>=0.22",
    ],
    extras_require={
        "dev": ["black", "isort", "flake8", "mypy", "pytest-cov"],
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,  # если есть не-Python файлы (шаблоны, статика)
    zip_safe=False,
)
