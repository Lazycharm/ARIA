FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        pydantic pydantic-settings \
        loguru apscheduler python-dotenv requests \
        plotly dash dash-bootstrap-components \
        pandas numpy scikit-learn lightgbm optuna \
        sqlalchemy aiosqlite ta \
        fastapi uvicorn websockets \
        httpx aiohttp tenacity arrow \
        anthropic praw yfinance pyarrow mlflow \
        python-telegram-bot \
        beautifulsoup4 lxml

# Copy source
COPY . .

# Create required directories
RUN mkdir -p db/backups logs

# Health check (Dash dashboard)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8050/ || exit 1

EXPOSE 8050 8051

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["python", "main.py", "--live"]
