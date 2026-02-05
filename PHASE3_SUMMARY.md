# Phase 3 Implementation Summary

## Overview
Phase 3 implements **LLM-driven agentic browsing** - an intelligent web navigation system that uses LLM to understand user prompts and navigate websites to collect relevant documents.

## Architecture

```
User Prompt → Gateway → BrowserUse Agent Service → Collected URLs → Scraper Service (Phase 1) → MinIO/Postgres
```

### Key Components

1. **BrowserUse Agent Service** (`scripts/phase3_agent_stub.py`)
   - LLM-guided navigation and URL collection
   - Supports both Ollama (local) and OpenAI (GPT-4o) providers
   - Intelligent link filtering based on user prompt
   - Document page detection vs category pages

2. **Gateway Demo** (`scripts/demo_phase3_gateway.py`)
   - Simulates gateway service
   - Converts collected URLs to Phase 1 config
   - Streams scraping progress

## Key Features Implemented

### 1. **LLM Provider Support**
- ✅ **Ollama** (local LLM) - Default
- ✅ **OpenAI** (GPT-4o, GPT-4, GPT-3.5-turbo)
- ✅ Environment variable configuration
- ✅ Seamless switching between providers

### 2. **Intelligent Navigation**
- ✅ **Keyword extraction** from user prompts
- ✅ **LLM-based link relevance** filtering
- ✅ **Document page detection** (vs category pages)
- ✅ **Strict relevance checking** (only collect if LLM says relevant AND is document page)

### 3. **Document vs Category Page Detection**
- ✅ Heuristic-based detection:
  - Text length > 800 chars
  - Has main content area (article/main tags)
  - Reasonable link-to-text ratio
  - Multiple substantial paragraphs
- ✅ Prevents collecting category pages that only list links

### 4. **Generic & Non-Hardcoded**
- ✅ No hardcoded website-specific logic
- ✅ Dynamic keyword extraction from prompts
- ✅ Generic prompts that work with any topic
- ✅ Works with any documentation website

### 5. **Debug Logging**
- ✅ Detailed agent navigation logs
- ✅ LLM decision tracking
- ✅ Link selection/filtering reasons
- ✅ Collection success/failure reasons

## Configuration

### Environment Variables (`env.example`)
```env
# LLM Provider
LLM_PROVIDER=ollama  # or "openai"

# Ollama
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_DEFAULT_MODEL=qwen2.5:3b

# OpenAI
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional
OPENAI_DEFAULT_MODEL=gpt-4o
```

### Optional: BrowserUse Backend
To use BrowserUse for Phase 3 URL collection, set:
```env
PHASE3_AGENT_BACKEND=browser_use
BROWSER_USE_SERVICE_BASE_URL=http://localhost:9010
BROWSER_USE_INTERNAL_API_KEY=
BROWSER_USE_TIMEOUT_SECONDS=600
BROWSER_USE_MAX_STEPS=60
BROWSER_USE_TEMPERATURE=0.2
BROWSER_USE_HEADLESS=true
```

### Command Line Usage

**With Ollama:**
```bash
python scripts/demo_phase3_gateway.py \
  --base-url https://www.cyon.ch/support \
  --prompt "find all the document with topic E-mail" \
  --llm-provider ollama \
  --max-depth 2 \
  --max-pages 10
```

**With OpenAI:**
```bash
python scripts/demo_phase3_gateway.py \
  --base-url https://www.cyon.ch/support \
  --prompt "find all the document with topic E-mail" \
  --llm-provider openai \
  --openai-model gpt-4o \
  --max-depth 2 \
  --max-pages 10
```

## Implementation Details

### 1. Keyword Extraction (`_extract_keywords_from_prompt`)
- Extracts topic keywords from user prompts
- Handles patterns like "topic X", "about X", "documents about X"
- Removes stop words
- Returns unique keywords for matching

### 2. Document Page Detection (`_is_likely_document_page`)
- Analyzes HTML structure
- Checks for article/main tags
- Calculates text-to-link ratio
- Counts substantial paragraphs
- Returns boolean + metadata

### 3. LLM Link Filtering (`_ask_llm_which_links_relevant`)
- Sends page content + available links to LLM
- Highlights links containing keywords
- LLM returns JSON array of relevant URLs
- Only follows links LLM deems relevant

### 4. LLM Relevance Checking (`_ask_llm_is_url_relevant`)
- Checks if a page is relevant to user's prompt
- Considers both relevance AND document page status
- Returns boolean decision
- Includes reasoning in response

### 5. Collection Logic
- **Strict filtering**: Only collect if `is_relevant AND is_document_page`
- Prevents collecting:
  - Category pages (even if relevant)
  - Irrelevant documents
  - Pages that don't match user's topic

## Files Created/Modified

### New Files
- `src/scraper/llm/openai_adapter.py` - OpenAI API adapter
- `scripts/phase3_agent_stub.py` - BrowserUse agent service
- `scripts/demo_phase3_gateway.py` - Gateway demo client
- `TEST_PHASE3_SETUP.md` - Setup guide
- `PHASE3_SUMMARY.md` - This file

### Modified Files
- `env.example` - Added OpenAI configuration
- `scripts/__init__.py` - Made scripts a package

## Testing Status

### Current Issues
- ⚠️ Missing dependencies (playwright, beautifulsoup4) - need installation
- ⚠️ Requires Ollama running OR OpenAI API key
- ⚠️ Requires scraper service running on port 50051

### Test Scenarios
1. ✅ Keyword extraction works
2. ✅ LLM provider switching works
3. ⏳ End-to-end navigation (needs dependencies)
4. ⏳ Document collection (needs dependencies)
5. ⏳ Relevance filtering (needs dependencies)

## Next Steps

1. **Install dependencies:**
   ```bash
   pip install playwright beautifulsoup4 openai
   python -m playwright install chromium
   ```

2. **Start services:**
   - Ollama: `ollama serve`
   - Scraper service: `uv run python -m scraper.main`

3. **Run test:**
   ```bash
   python scripts/demo_phase3_gateway.py \
     --base-url https://www.cyon.ch/support \
     --prompt "find all the document with topic E-mail" \
     --max-depth 2 \
     --max-pages 5
   ```

## Key Improvements Made

1. **Removed hardcoding**: Generic prompts, no website-specific logic
2. **Better filtering**: Strict AND logic (relevant + document page)
3. **Debug logging**: Comprehensive logs for troubleshooting
4. **Provider flexibility**: Easy switching between Ollama and OpenAI
5. **Error handling**: Graceful fallbacks and clear error messages

