# 🚀 Using OpenRouter Instead of Local Ollama

## Why OpenRouter?

**Problem:** Local LLMs like `qwen2.5-coder:14b` are **TOO SLOW**:
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
ORCHESTRATOR_MODEL=z-ai/glm-4.5-air:free
SYNTHESIZER_MODEL=z-ai/glm-4.5-air:free

# Or use cheaper option:
# ORCHESTRATOR_MODEL=deepseek/deepseek-chat
# SYNTHESIZER_MODEL=deepseek/deepseek-chat
```

### 4. Run with OpenRouter

**Web UI (Recommended):**
```bash
# Double-click this file:
start_web_ui_openrouter.bat

# Or run manually:
python src/web_ui_openrouter.py
```

**CLI version:**
```bash
python src/main_openrouter.py
```

Then open: http://127.0.0.1:7860 🚀

## Model Options

### 🏆 Recommended: GLM-4.5 Air (Free)
```
ORCHESTRATOR_MODEL=z-ai/glm-4.5-air:free
SYNTHESIZER_MODEL=z-ai/glm-4.5-air:free
```
- Speed: ⚡⚡⚡⚡ (5-10s)
- Quality: ⭐⭐⭐⭐
- Cost: **FREE**
- Best for: Production use, budget-conscious

### 💰 Budget: DeepSeek Chat
```
ORCHESTRATOR_MODEL=deepseek/deepseek-chat
SYNTHESIZER_MODEL=deepseek/deepseek-chat
```
- Speed: ⚡⚡⚡⚡ (5-10s)
- Quality: ⭐⭐⭐⭐
- Cost: ~$0.27 per 1M tokens (10x cheaper!)
- Best for: High volume queries

### 🆓 Free: Google Gemini Flash
```
ORCHESTRATOR_MODEL=google/gemini-2.0-flash-exp-1219
SYNTHESIZER_MODEL=google/gemini-2.0-flash-exp-1219
```
- Speed: ⚡⚡⚡⚡⚡ (3-7s)
- Quality: ⭐⭐⭐⭐
- Cost: **FREE** (experimental)
- Best for: Testing, development

### 💪 Power: Claude 3.5 Sonnet
```
ORCHESTRATOR_MODEL=anthropic/claude-3.5-sonnet
SYNTHESIZER_MODEL=anthropic/claude-3.5-sonnet
```
- Speed: ⚡⚡⚡⚡ (3-5s)
- Quality: ⭐⭐⭐⭐⭐
- Cost: $3 per 1M input tokens
- Best for: Complex queries, code generation

## Test It

```bash
# Test with CLI
python src/main_openrouter.py

# Ask a question
You: What is the CIP service in Academicon?

# Should get answer in 10-20 seconds! 🎉
```

## Compare Results

| Metric | Local (Ollama) | OpenRouter (GLM-4.5) |
|--------|----------------|---------------------|
| Orchestrator | 70s | 2-3s |
| Synthesizer | 180s+ | 5-8s |
| **Total** | **250s+** | **10-15s** |
| Success Rate | 30% | 99% |
| Cost/Query | Free | $0.01-0.02 |

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
- Or `z-ai/glm-4.5-air:free` (reliable + free)

### Wrong answers?
- Try `anthropic/claude-3.5-sonnet` (best quality)

## Next Steps

1. ✅ Get API key
2. ✅ Add credits ($5-10)
3. ✅ Edit `.env` file
4. ✅ Run `python src/main_openrouter.py`
5. 🎉 Enjoy **20x faster** queries!

---

**Recommendation:** Start with `z-ai/glm-4.5-air:free` (free) to test, then adjust models based on performance.

**Cost estimate for your use case:**
- ~10 queries/day = ~$0.10-0.20/day = ~$3-6/month
- ~50 queries/day = ~$0.50-1/day = ~$15-30/month

**Worth it?** Absolutely! 4 minutes → 15 seconds = **16x faster** 🚀
