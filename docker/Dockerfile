FROM python:3.11-slim

WORKDIR /app

# System deps for Playwright (Chromium) + WeasyPrint
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libffi-dev \
    libcairo2 \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    shared-mime-info \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml README.md ./ 

# Install dependencies
RUN uv pip install --system -e .

# Install Playwright browsers + deps
RUN python -m playwright install --with-deps chromium

# Copy application code
COPY . .

# Generate gRPC code
RUN python scripts/generate_proto.py

ENV PYTHONPATH=/app:/app/src

EXPOSE 50051

CMD ["python", "-m", "scraper.main"]


