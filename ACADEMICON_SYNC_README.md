# Academicon Auto-Sync System

Αυτό το σύστημα αυτοματοποιεί το sync μεταξύ του Academicon project στο WSL Ubuntu και του αντίγραφου στο Windows D: drive.

## 📋 Προαπαιτούμενα

1. **WSL Ubuntu** με το Academicon project στο: `/home/deeznutz/projects/Academicon-Rebuild`
2. **Windows D: drive** με αντίγραφο στο: `D:\Academicon-Rebuild`
3. **Git** εγκατεστημένο σε WSL και Windows
4. **GitHub repository** ως authoritative source: `https://github.com/Deefunxion/Academicon-Web`

## 🚀 Γρήγορη Εγκατάσταση

### Βήμα 1: Setup Git Remote στο WSL
```bash
# Στο WSL Ubuntu terminal:
cd /home/deeznutz/projects/Academicon-Rebuild
git remote add windows /mnt/d/Academicon-Rebuild
git branch -M main
```

### Βήμα 2: Αρχικό Push στο Windows
```bash
# Στο WSL Ubuntu terminal:
git push windows main
```

### Βήμα 3: Setup Automated Sync
```powershell
# Στο Windows PowerShell (ως Administrator):
cd D:\LOCAL-CODER
.\setup_sync_task.ps1
```

## 📁 Αρχεία

- `sync_academicon.bat` - Απλό batch script για manual sync
- `sync_academicon.ps1` - Προηγμένο PowerShell script με logging
- `setup_sync_task.ps1` - Setup script για Windows Task Scheduler
- `sync_log.txt` - Log file για το PowerShell script

## 🔧 Χρήση

### Manual Sync
```batch
# Κάνε double-click στο sync_academicon.bat
# ή τρέξε στο command prompt:
D:\LOCAL-CODER\sync_academicon.bat
```

### Automated Sync
Το σύστημα τρέχει αυτόματα κάθε 30 λεπτά μέσω Windows Task Scheduler.

### Force Sync με Verbose Output
```powershell
cd D:\LOCAL-CODER
.\sync_academicon.ps1 -Force -Verbose
```

## 📊 Monitoring

- **Logs**: `D:\LOCAL-CODER\sync_log.txt`
- **Task Status**: Task Scheduler → Task Scheduler Library → "Academicon Auto-Sync"
- **Last Run**: Check το log file ή Task Scheduler history

## 🔄 Πώς Δουλεύει

1. **WSL Changes**: Όλες οι αλλαγές γίνονται στο WSL project
2. **Auto-Commit**: Το script κάνει commit των αλλαγών στο WSL
3. **Push to Windows**: Push στο Windows αντίγραφο μέσω git
4. **Indexing**: Το D:\ αντίγραφο είναι έτοιμο για γρήγορο indexing

## 🛠 Troubleshooting

### Sync Αποτυγχάνει
1. Check ότι το WSL Ubuntu τρέχει
2. Verify ότι τα paths είναι σωστά
3. Check το `sync_log.txt` για errors
4. Manual run: `.\sync_academicon.ps1 -Verbose`

### Task Scheduler Issues
1. Run `setup_sync_task.ps1` ως Administrator
2. Check Task Scheduler για errors
3. Verify ότι το PowerShell script υπάρχει

### Git Issues
```bash
# Στο WSL - check remotes:
git remote -v

# Force push αν χρειάζεται:
git push windows main --force
```

## ⚙️ Customization

### Αλλαγή Sync Frequency
Edit το `setup_sync_task.ps1` και άλλαξε το `-RepetitionInterval`:

```powershell
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 15) -RepetitionDuration (New-TimeSpan -Days 1)
```

### Αλλαγή Paths
Update τα paths στα scripts αν χρειάζεται.

## 📞 Support

Για προβλήματα:
1. Check τα logs
2. Verify WSL connection
3. Test manual sync
4. Check GitHub repository status