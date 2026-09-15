FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

COPY pyproject.toml README.md ./
COPY src ./src
COPY knowledge ./knowledge
COPY config ./config

RUN pip install --no-cache-dir ".[server,observability]"

EXPOSE 8790

CMD ["message-platform-helper", "serve-fastapi", "--host", "0.0.0.0", "--port", "8790"]
