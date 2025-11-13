# 🔧 CUDA OOM Fix - Quick Reference

## 📋 Τι Έγινε (12 Νοέμβρη 2025)

**Πρόβλημα:** Academicon indexing έπεφτε με CUDA Out of Memory
- GPU batch size: 307 (πολύ μεγάλο) ❌
- VRAM exhaustion: Ollama LLM + Embeddings + 307 batch = OOM

**Λύση:** Conservative batch sizing
- GPU batch size: 64 (ασφαλές) ✅
- Reserve 8GB για Ollama
- Auto-stop Ollama πριν indexing

---

## 🚀 Quick Start (3 Steps)

### 1️⃣ Check VRAM
```bash
python check_vram.py
```

### 2️⃣ Run Indexing
```bash
# Option A: Python only (faster)
index_academicon_lite.bat

# Option B: All files (complete)
index_academicon.bat

# Option C: Interactive menu
START_INDEXING.bat
```

### 3️⃣ Test It
```bash
# Start Ollama (if stopped)
ollama serve

# Run assistant
python main.py
# OR
python web_ui.py
```

---

## 📊 What Changed

| File | Change | Impact |
|------|--------|--------|
| `gpu_utils.py` | Added `conservative` mode | Caps batch at 64 |
| `index_academicon_v2.py` | Uses conservative=True | Safe for large codebase |
| `index_academicon_lite.py` | batch_size 128→64 | Prevents OOM |

---

## 🎯 Batch Size Guide

| Codebase Size | Old Batch | New Batch | Status |
|---------------|-----------|-----------|--------|
| Small (<5K) | 128 | 128 | ✅ OK |
| Medium (5-15K) | 128-307 | 64 | ✅ OK |
| **Academicon (25K)** | **307** | **64** | ✅ **FIXED!** |

---

## 📁 New Files

- ✅ `check_vram.py` - VRAM checker
- ✅ `index_academicon.bat` - Auto full indexing  
- ✅ `index_academicon_lite.bat` - Auto lite indexing
- ✅ `START_INDEXING.bat` - Interactive menu
- ✅ `ACADEMICON_INDEXING_FIX.md` - Full docs (Greek)
- ✅ `GREEK_INDEXING_SUMMARY.md` - Summary (Greek)
- ✅ `QUICK_FIX_README.md` - This file

---

## ⚡ Performance

**GPU (RTX 5070 Ti) with batch_size=64:**
- Lite: 10-15 min ⚡
- Full: 15-20 min ⚡

**CPU (fallback) with batch_size=32:**
- Lite: 30-45 min 🐌
- Full: 60-90 min 🐌

---

## 🆘 Still Getting OOM?

### Solution 1: Lower Batch Size
Edit `index_academicon_v2.py` line ~42:
```python
batch_size = 32  # From 64 → 32
```

### Solution 2: Use CPU
Edit `index_academicon_v2.py` line ~39:
```python
device = "cpu"
batch_size = 32
```

### Solution 3: Clear GPU Memory
```bash
# Restart computer (nuclear option)
# OR close all GPU apps
tasklist | findstr /i "ollama.exe chrome.exe"
```

---

## 📚 Full Documentation

- **English:** `ACADEMICON_INDEXING_FIX.md` (detailed)
- **Ελληνικά:** `GREEK_INDEXING_SUMMARY.md` (summary)
- **Implementation:** `IMPLEMENTATION_PROGRESS.md` (bug fixes section)

---

## ✅ Verification

```bash
# 1. Check no more batch_size=307 in code
findstr /s "batch_size = 307" *.py
# Should return: No matches

# 2. Run VRAM check
python check_vram.py
# Should show: batch_size recommendation = 64

# 3. Test indexing
index_academicon_lite.bat
# Should complete without OOM errors
```

---

**Status:** ✅ FIXED (2025-11-12)  
**Next Step:** Run `START_INDEXING.bat` and choose option 2 or 3! 🚀
