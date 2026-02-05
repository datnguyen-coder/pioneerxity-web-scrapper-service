FROM mcr.microsoft.com/playwright/python:v1.53.0-noble

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
RUN uv venv --python 3.11
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN uv pip install -e .

EXPOSE 9010

CMD ["python", "-m", "browser_use_service.main"]


