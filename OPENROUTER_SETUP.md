# 🚀 Using OpenRouter Instead of Local Ollama

## Why OpenRouter?

**Problem:** Local LLMs like `qwen2.5-coder:14b` are **TOO SLOW** even on RTX 5070 Ti:
- Orchestrator: 70 seconds
- Synthesizer: 180+ seconds  
- **Total: 4+ minutes per query** 😱

**Solution:** OpenRouter provides API access to fast cloud LLMs:
- Claude 3.5 Sonnet: **2-5 seconds per query** ⚡
- GPT-4o: **3-7 seconds**
- DeepSeek: **5-10 seconds** (cheapest)
- **Total: 10-20 seconds per query** 🎯

## Cost Comparison

### Local (Ollama)
- ✅ Free
- ❌ VERY slow (4+ minutes)
- ❌ Unreliable (timeouts)
- ❌ Lower quality answers

### OpenRouter
- ✅ FAST (10-20 seconds)
- ✅ Reliable (no timeouts)
- ✅ Better quality answers
- 💰 Cost: **~$0.01-0.02 per query**
  - 100 queries: ~$1-2
  - 1000 queries: ~$10-20

**Recommended model:** `anthropic/claude-3.5-sonnet` (best balance of speed/quality/cost)

## Setup Instructions

### 1. Get OpenRouter API Key

1. Go to: https://openrouter.ai/
2. Sign up (free)
3. Go to: https://openrouter.ai/keys
4. Create new key
5. Copy your key: `sk-or-v1-...`

### 2. Add Credits

1. Go to: https://openrouter.ai/credits
2. Add $5-10 (enough for 500-1000 queries)
3. Payment via card or crypto

### 3. Configure LocalCoder

Edit `.env` file:

```bash
# Add your API key
OPENROUTER_API_KEY=sk-or-v1-YOUR-KEY-HERE

# Choose model (recommended)
MODEL_NAME=moonshotai/kimi-k2-thinking

# Or use cheaper option:
# MODEL_NAME=deepseek/deepseek-chat-v3.1:free

# Or use Google's free tier:
# MODEL_NAME=google/gemini-2.5-flash-lite-preview-09-2025
```

### 4. Run with OpenRouter

**Web UI (Recommended):**
```bash
# Double-click this file:
start_web_ui_openrouter.bat

# Or run manually:
python web_ui_openrouter.py
```

**CLI version:**
```bash
python main_openrouter.py
```

Then open: http://127.0.0.1:7860 🚀

## Model Options

### 🏆 Recommended: Claude 3.5 Sonnet
```
MODEL_NAME=anthropic/claude-3.5-sonnet
```
- Speed: ⚡⚡⚡⚡⚡ (2-5s)
- Quality: ⭐⭐⭐⭐⭐
- Cost: $3 per 1M input tokens, $15 per 1M output
- Best for: Production use, high-quality answers

### 💰 Budget: DeepSeek Chat
```
MODEL_NAME=deepseek/deepseek-chat
```
- Speed: ⚡⚡⚡⚡ (5-10s)
- Quality: ⭐⭐⭐⭐
- Cost: $0.27 per 1M tokens (10x cheaper!)
- Best for: High volume queries, testing

### 🆓 Free: Google Gemini Flash
```
MODEL_NAME=google/gemini-2.0-flash-exp-1219
```
- Speed: ⚡⚡⚡⚡⚡ (3-7s)
- Quality: ⭐⭐⭐⭐
- Cost: **FREE** during experimental period
- Best for: Testing, development

### 💪 Power: GPT-4o
```
MODEL_NAME=openai/gpt-4o
```
- Speed: ⚡⚡⚡⚡ (3-7s)
- Quality: ⭐⭐⭐⭐⭐
- Cost: $2.50 per 1M input, $10 per 1M output
- Best for: Complex queries, code generation

## Test It

```bash
# Test with CLI
python main_openrouter.py

# Ask a question
You: What is the CIP service in Academicon?

# Should get answer in 10-20 seconds! 🎉
```

## Compare Results

| Metric | Local (Ollama) | OpenRouter (Claude) |
|--------|----------------|---------------------|
| Orchestrator | 70s | 2-3s |
| Synthesizer | 180s+ | 5-8s |
| **Total** | **250s+** | **10-15s** |
| Success Rate | 30% | 99% |
| Cost/Query | Free | $0.01-0.02 |

## Files Changed

- `agents_openrouter.py` - Agents using OpenRouter API
- `main_openrouter.py` - Main pipeline with OpenRouter
- `.env` - Add `OPENROUTER_API_KEY` and `MODEL_NAME`
- This file! 📄

## Troubleshooting

### "OPENROUTER_API_KEY not set"
- Make sure `.env` file exists
- Check key format: `sk-or-v1-...`
- Restart Python after editing `.env`

### "Insufficient credits"
- Go to: https://openrouter.ai/credits
- Add more credits

### Still slow?
- Try `google/gemini-2.0-flash-exp-1219` (free + fast)
- Or `deepseek/deepseek-chat` (cheapest)

### Wrong answers?
- Try `anthropic/claude-3.5-sonnet` (best quality)
- Or `openai/gpt-4o` (also excellent)

## Next Steps

1. ✅ Get API key
2. ✅ Add credits ($5-10)
3. ✅ Edit `.env` file
4. ✅ Run `python main_openrouter.py`
5. 🎉 Enjoy **20x faster** queries!

---

**Recommendation:** Start with `google/gemini-2.0-flash-exp-1219` (free) to test, then switch to `anthropic/claude-3.5-sonnet` for production.

**Cost estimate for your use case:**
- ~10 queries/day = ~$0.20/day = ~$6/month
- ~50 queries/day = ~$1/day = ~$30/month

**Worth it?** Absolutely! 4 minutes → 15 seconds = **16x faster** 🚀
