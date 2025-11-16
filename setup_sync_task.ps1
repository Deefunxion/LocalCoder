# Setup Windows Task Scheduler for automatic Academicon sync
# Run this script as Administrator to create the scheduled task

$TaskName = "Academicon Auto-Sync"
$ScriptPath = "D:\LOCAL-CODER\sync_academicon.ps1"
$LogFile = "D:\LOCAL-CODER\task_scheduler_setup.log"

function Write-Log {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogMessage = "[$Timestamp] $Message"
    Write-Host $LogMessage
    Add-Content -Path $LogFile -Value $LogMessage
}

function Test-AdminPrivileges {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Create-ScheduledTask {
    Write-Log "Creating scheduled task: $TaskName"

    # Check if task already exists
    $existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existingTask) {
        Write-Log "Task already exists. Removing old task..."
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }

    # Create new task
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File `"$ScriptPath`" -Verbose"

    # Run every 30 minutes when logged in
    $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 30) -RepetitionDuration (New-TimeSpan -Days 1)

    # Run with highest privileges
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest

    # Create settings
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RunOnlyIfNetworkAvailable

    # Register the task
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "Automatically sync Academicon project from WSL to Windows D: drive"

    Write-Log "Scheduled task created successfully"
}

# Main execution
try {
    Write-Log "=========================================="
    Write-Log "Setting up Academicon Auto-Sync Task"
    Write-Log "=========================================="

    # Check admin privileges
    if (-not (Test-AdminPrivileges)) {
        Write-Log "ERROR: This script must be run as Administrator"
        Write-Log "Please right-click and select Run as administrator"
        exit 1
    }

    Write-Log "Running with administrator privileges"

    # Check if PowerShell script exists
    if (-not (Test-Path $ScriptPath)) {
        Write-Log "ERROR: PowerShell script not found at $ScriptPath"
        exit 1
    }

    Write-Log "PowerShell script found"

    # Create the scheduled task
    Create-ScheduledTask

    Write-Log "=========================================="
    Write-Log "SUCCESS: Scheduled task setup completed"
    Write-Log "=========================================="
    Write-Log ""
    Write-Log "The task $TaskName has been created and will run:"
    Write-Log "  - Every 30 minutes"
    Write-Log "  - When you are logged in"
    Write-Log "  - With highest privileges"
    Write-Log "  - Only when network is available"
    Write-Log ""
    Write-Log "You can manage this task in Task Scheduler:"
    Write-Log "  1. Open Task Scheduler"
    Write-Log "  2. Go to Task Scheduler Library"
    Write-Log "  3. Find $TaskName"
    Write-Log ""

}
catch {
    $errorMessage = $_.Exception.Message
    $errorType = $_.Exception.GetType().Name
    Write-Log "CRITICAL ERROR: $errorType - $errorMessage"
    Write-Log "Stack Trace: $($_.Exception.StackTrace)"
    exit 1
}