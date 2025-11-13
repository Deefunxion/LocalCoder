# 🚀 Πως να Χρησιμοποιήσεις τον Academicon Code Assistant

## Γρήγορη Εκκίνηση

### Τρόπος 1: Web UI (Συνιστάται) 🌐

**Πιο εύκολος τρόπος - Όμορφο interface στο browser!**

1. Κάνε διπλό κλικ στο: `start_web_ui.bat`
2. Περίμενε 10-20 δευτερόλεπτα να φορτώσει
3. Θα ανοίξει το browser αυτόματα στο `http://localhost:7860`
4. Γράψε την ερώτησή σου και πάτα "Send"!

**Για να σταματήσεις:**
- Πάτα `Ctrl+C` στο παράθυρο που άνοιξε
- Ή απλά κλείσε το παράθυρο

---

### Τρόπος 2: Command Line (CLI) 💻

**Για πιο advanced χρήση:**

1. Άνοιξε Git Bash ή PowerShell
2. Πήγαινε στον φάκελο:
   ```bash
   cd D:\LOCAL-CODER
   ```

3. Ενεργοποίησε το environment:
   ```bash
   academicon-agent-env/Scripts/activate
   ```

4. Τρέξε τον assistant:
   ```bash
   python main.py
   ```

5. Γράψε τις ερωτήσεις σου:
   ```
   You: What is the CIP service?
   [Περιμένεις 10-20 δευτερόλεπτα]
   Assistant: [Απάντηση εδώ]
   ```

6. Για έξοδο γράψε: `exit` ή `quit`

---

## Παραδείγματα Ερωτήσεων

### Γενικές Ερωτήσεις
- "What is the CIP service in Academicon?"
- "What is the project structure?"
- "What libraries are being used?"

### Specific Features
- "How does user authentication work?"
- "Show me the database models for publications"
- "What API endpoints are available?"
- "How is the task queue implemented?"

### Code Search
- "Find all functions related to citations"
- "Show me where email notifications are sent"
- "How are PDFs generated?"

### Technical Questions
- "What's the error handling strategy?"
- "How is data validated?"
- "What testing framework is used?"

---

## Τι Μπορεί να Κάνει

✅ **Αναζήτηση κώδικα**: Βρίσκει relevant code chunks
✅ **Εξήγηση**: Εξηγεί πως δουλεύουν functions/classes
✅ **Σχέσεις**: Αναλύει dependencies και relationships
✅ **Documentation**: Απαντάει σε architectural questions

❌ **ΔΕΝ μπορεί να:**
- Τρέξει κώδικα
- Κάνει αλλαγές στα αρχεία
- Συνδεθεί στο internet για updates

---

## Απόδοση

**Χρόνος απόκρισης**: 10-20 δευτερόλεπτα ανά ερώτηση

**Τι τρέχει:**
1. Orchestrator (1-3s): Σχεδιάζει την αναζήτηση
2. Indexer (0.5-1s): Βρίσκει relevant code
3. Graph Analyst (2-5s): Αναλύει σχέσεις
4. Synthesizer (5-15s): Δημιουργεί την απάντηση

---

## Troubleshooting

### "Cannot connect" ή δεν ανοίγει το UI
- Σιγουρέψου ότι το Ollama τρέχει: `ollama list`
- Restart το web UI: Κλείσε και ξανάνοιξε το `start_web_ui.bat`

### Πολύ αργές απαντήσεις
- Κλείσε άλλες εφαρμογές που τρώνε RAM
- Το πρώτο query είναι πιο αργό (φορτώνει models)

### "Model not found"
- Τρέξε: `ollama list` για να δεις τα models
- Αν λείπει το qwen2.5-coder:14b, τρέξε:
  ```bash
  ollama pull qwen2.5-coder:14b
  ```

### Database error
- Αν έσβησες το `academicon_chroma_db/`, τρέξε:
  ```bash
  python index_academicon_lite.py
  ```
  Θα πάρει 5-10 λεπτά

---

## Αρχεία & Φάκελοι

```
D:\LOCAL-CODER\
├── start_web_ui.bat          ← Διπλό κλικ εδώ για Web UI!
├── web_ui.py                 ← Web interface code
├── main.py                   ← CLI interface
├── agents.py                 ← AI agents definitions
├── index_academicon_lite.py  ← Indexing script
├── academicon_chroma_db/     ← Vector database (77MB)
├── academicon-agent-env/     ← Python environment
└── HOW_TO_USE.md            ← Αυτό το αρχείο!
```

---

## Tips

💡 **Γράψε ξεκάθαρες ερωτήσεις**
- Καλό: "How does the CIP service calculate scores?"
- Όχι καλό: "CIP scores"

💡 **Αν δεν βρίσκει κάτι**
- Δοκίμασε διαφορετική διατύπωση
- Ρώτα πιο συγκεκριμένα

💡 **Για γρήγορες απαντήσεις**
- Χρησιμοποίησε το Web UI
- Κράτα το παράθυρο ανοιχτό (δεν χρειάζεται restart)

---

## Συντήρηση

### Re-indexing (Αν αλλάξει ο κώδικας)
```bash
cd D:\LOCAL-CODER
academicon-agent-env/Scripts/activate
python index_academicon_lite.py
```
Χρόνος: 5-10 λεπτά

### Model Updates
```bash
ollama pull qwen2.5-coder:14b
```

---

## Support

Αν χρειαστείς βοήθεια:
1. Διάβασε το `README.md`
2. Τσέκαρε το `CLAUDE.md` για technical details
3. Δοκίμασε τα test scripts: `python quick_test.py`

---

**Καλή χρήση! 🚀**
