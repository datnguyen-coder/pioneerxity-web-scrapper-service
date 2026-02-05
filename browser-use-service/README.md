 # BrowserUse Service (POC)
 
 FastAPI microservice that runs a BrowserUse agent and streams progress via SSE.
 
 ## Endpoints
 - `GET /health`
 - `POST /api/v1/browser-use/run/stream` (SSE)
 
 ## Environment
 - `BROWSER_USE_INTERNAL_API_KEY` (shared secret for `X-Gateway-Key`)
 - `OPENAI_API_KEY` (if using OpenAI)
 - `OPENAI_BASE_URL` (optional)
 - `OLLAMA_ENDPOINT` (optional, default `http://localhost:11434`)
 
 ## Run (local)
 ```bash
 uv venv --python 3.11
 .venv\Scripts\Activate.ps1
 uv pip install -e .
 uv run python -m browser_use_service.main
 ```
 

