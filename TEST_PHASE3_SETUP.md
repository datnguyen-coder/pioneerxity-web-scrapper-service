# Phase 3 Demo - Setup & Test Guide

## Prerequisites Checklist

### 1. ✅ Ollama LLM Running
- **Required**: Ollama must be running with a model installed
- **Default**: `qwen2.5:3b` on `localhost:11434`

**Check if Ollama is running:**
```bash
curl http://localhost:11434/api/tags
```

**If not running, start Ollama:**
```bash
# Install Ollama if needed: https://ollama.ai
ollama serve
```

**Pull the model (if not already installed):**
```bash
ollama pull qwen2.5:3b
```

### 2. ✅ Scraper Service (gRPC) Running
- **Required**: Scraper service must be running on port 50051
- **Default**: `localhost:50051`

**Start the scraper service:**
```bash
# In a separate terminal
uv run python -m scraper.main
```

**Or run in background:**
```bash
uv run python -m scraper.main > server-50051.log 2>&1 &
```

**Verify it's running:**
```bash
# Should see gRPC server listening
# Or check logs: tail -f server-50051.log
```

### 3. ✅ Dependencies Installed
- **Required**: All Python dependencies must be installed

**Install dependencies:**
```bash
# Activate your virtual environment first
uv pip install -e .

# Or install specific packages if needed:
uv pip install playwright beautifulsoup4 grpcio grpcio-tools
python -m playwright install chromium
```

### Optional: BrowserUse Service (Phase 3 backend)
- If you want BrowserUse-driven navigation, start the BrowserUse service:
```bash
cd browser-use-service
uv venv --python 3.11
.venv\Scripts\Activate.ps1
uv pip install -e .
uv run python -m browser_use_service.main
```
- Then set `PHASE3_AGENT_BACKEND=browser_use` in `.env`.

### 4. ✅ Proto Files Generated
- **Required**: gRPC proto files must be generated

**Generate proto files:**
```bash
uv run python scripts/generate_proto.py
```

### 5. ✅ Environment Variables (Optional)
- **Note**: For full functionality, you may need `.env` file with MinIO/Postgres config
- **For demo**: Can run without if service has defaults

**Create `.env` from template:**
```bash
cp env.example .env
# Edit .env with your MinIO/Postgres credentials if needed
```

---

## Running the Test

### Step 1: Start Ollama (if not running)
```bash
ollama serve
```

### Step 2: Start Scraper Service (in separate terminal)
```bash
uv run python -m scraper.main
```

### Step 3: Run Phase 3 Demo

**For PowerShell (Windows):**
```powershell
python scripts/demo_phase3_gateway.py --base-url https://www.cyon.ch/support --prompt "find all the document with topic E-mail" --max-depth 2 --max-pages 10 --output-format pdf --include-images --image-handling embed_or_link --ollama-host localhost --ollama-port 11434 --ollama-model qwen2.5:3b
```

**For Bash/Linux/Mac:**
```bash
python scripts/demo_phase3_gateway.py \
  --base-url https://www.cyon.ch/support \
  --prompt "find all the document with topic E-mail" \
  --max-depth 2 \
  --max-pages 10 \
  --output-format pdf \
  --include-images \
  --image-handling embed_or_link \
  --ollama-host localhost \
  --ollama-port 11434 \
  --ollama-model qwen2.5:3b
```

### Alternative: Minimal Test (fewer pages)

**PowerShell:**
```powershell
python scripts/demo_phase3_gateway.py --base-url https://www.cyon.ch/support --prompt "find all the document with topic E-mail" --max-depth 1 --max-pages 5
```

**Bash:**
```bash
python scripts/demo_phase3_gateway.py \
  --base-url https://www.cyon.ch/support \
  --prompt "find all the document with topic E-mail" \
  --max-depth 1 \
  --max-pages 5
```

---

## Expected Output

1. **Agent Phase**: LLM navigates and collects relevant URLs
   - You'll see the agent visiting pages
   - LLM filtering links based on prompt

2. **Scraping Phase**: Scraper service processes collected URLs
   - Progress updates streamed
   - Documents generated and uploaded to MinIO

3. **Final Stats**: Summary of processed documents

---

## Troubleshooting

### Error: "gRPC code not generated"
```bash
uv run python scripts/generate_proto.py
```

### Error: "Connection refused" (Ollama)
- Check if Ollama is running: `curl http://localhost:11434/api/tags`
- Start Ollama: `ollama serve`

### Error: "Connection refused" (Scraper Service)
- Check if service is running on port 50051
- Start service: `uv run python -m scraper.main`

### Error: "ModuleNotFoundError"
- Install dependencies: `uv pip install -e .`
- Or specific package: `uv pip install <package-name>`

### Error: "playwright not found"
```bash
uv pip install playwright
python -m playwright install chromium
```

---

## Quick Test Command (All-in-One)

If everything is set up, just run:

**PowerShell:**
```powershell
python scripts/demo_phase3_gateway.py --base-url https://www.cyon.ch/support --prompt "find all the document with topic E-mail" --max-depth 2 --max-pages 10
```

**Bash:**
```bash
python scripts/demo_phase3_gateway.py \
  --base-url https://www.cyon.ch/support \
  --prompt "find all the document with topic E-mail" \
  --max-depth 2 \
  --max-pages 10
```

