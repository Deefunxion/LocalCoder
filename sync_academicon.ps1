# Auto-sync Academicon-Rebuild from WSL to Windows D: drive
# Advanced PowerShell script with error handling and logging

param(
    [switch]$Force,
    [switch]$Verbose
)

$LogFile = "D:\LOCAL-CODER\sync_log.txt"
$WSLPath = "\\wsl.localhost\Ubuntu\home\deeznutz\projects\Academicon-Rebuild"
$WindowsPath = "D:\Academicon-Rebuild"

function Write-Log {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogMessage = "[$Timestamp] $Message"
    Write-Host $LogMessage
    Add-Content -Path $LogFile -Value $LogMessage
}

function Test-WSLConnection {
    try {
        $result = wsl -d Ubuntu -u deeznutz -- echo "WSL connection test"
        return $true
    }
    catch {
        return $false
    }
}

function Sync-Academicon {
    Write-Log "=========================================="
    Write-Log "🔄 Starting Academicon Auto-Sync"
    Write-Log "=========================================="

    # Test WSL connection
    if (-not (Test-WSLConnection)) {
        Write-Log "❌ ERROR: Cannot connect to WSL Ubuntu"
        return $false
    }

    Write-Log "✅ WSL connection established"

    # Check if WSL project exists
    $wslExists = wsl -d Ubuntu -u deeznutz -- test -d "/home/deeznutz/projects/Academicon-Rebuild" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Log "❌ ERROR: WSL project directory not found"
        return $false
    }

    Write-Log "✅ WSL project directory found"

    # Commit changes in WSL
    Write-Log "📝 Committing changes in WSL..."
    $commitResult = wsl -d Ubuntu -u deeznutz -- bash -c "cd /home/deeznutz/projects/Academicon-Rebuild && git add . && git commit -m 'Auto-sync: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')' --allow-empty" 2>&1

    if ($Verbose) {
        Write-Log "Commit output: $commitResult"
    }

    # Push to Windows copy
    Write-Log "🔄 Pushing to Windows drive..."
    $pushResult = wsl -d Ubuntu -u deeznutz -- bash -c "cd /home/deeznutz/projects/Academicon-Rebuild && git push windows main" 2>&1

    if ($Verbose) {
        Write-Log "Push output: $pushResult"
    }

    if ($LASTEXITCODE -eq 0) {
        Write-Log "✅ SUCCESS: Sync completed successfully"
        Write-Log "The D:\Academicon-Rebuild folder is now up-to-date"
        return $true
    } else {
        Write-Log "❌ ERROR: Sync failed"
        Write-Log "Push output: $pushResult"
        return $false
    }
}

# Main execution
try {
    if ($Force -or $Verbose) {
        Write-Log "Running with parameters: Force=$Force, Verbose=$Verbose"
    }

    $success = Sync-Academicon

    Write-Log "=========================================="
    if ($success) {
        Write-Log "🎉 Sync operation completed successfully"
    } else {
        Write-Log "💥 Sync operation failed - check log for details"
    }
    Write-Log "=========================================="

    exit [int](-not $success)
}
catch {
    Write-Log "💥 CRITICAL ERROR: $($_.Exception.Message)"
    exit 1
}