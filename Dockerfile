FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt constraints.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY backend ./backend
COPY demo ./demo
COPY docs ./docs
COPY experiments ./experiments
COPY frontend ./frontend
COPY src ./src
COPY README.md .

EXPOSE 8000 8501

CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
