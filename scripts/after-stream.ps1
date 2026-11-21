# SPHURA: turn the stream you just finished into Shorts.
# Run via "After Stream.bat" (double-click) once OBS has stopped recording.
#   after-stream.ps1 [-File <video>] [-Styles meme,cinematic] [-Game "Name"] [-Top 4]

param(
    [string]$File = "",
    [string[]]$Styles = @("meme", "cinematic", "hype"),
    [string]$Game = "",
    [int]$Top = 4,
    [string]$OutRoot = (Join-Path $env:USERPROFILE "Videos\SPHURA")
)

# "Continue": in Windows PowerShell, "Stop" turns any stderr line from a program into a crash
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root "engine\.venv\Scripts\python.exe"
$StateDir = Join-Path $env:LOCALAPPDATA "SPHURA"
$VideoExt = @(".mp4", ".mkv", ".mov", ".flv", ".ts")

function Say($text, $color = "Cyan") { Write-Host $text -ForegroundColor $color }
function Fail($text) {
    Write-Host "`n$text" -ForegroundColor Red
    exit 1
}

# --- Keep the PC awake while this runs (released automatically when the script exits)
Add-Type -Namespace CF -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
'@
[void][CF.Power]::SetThreadExecutionState([uint32]2147483649)  # CONTINUOUS | SYSTEM_REQUIRED

if (-not (Test-Path $Py)) { Fail "SPHURA environment not found at $Py" }
function Invoke-Sphura { & $Py -m sphura @args }  # works wherever the folder lives

# --- Where does OBS save recordings? Read it from OBS's own settings
function Get-ObsRecordingFolder {
    $obs = Join-Path $env:APPDATA "obs-studio"
    $profileDir = $null
    foreach ($ini in @("user.ini", "global.ini")) {   # OBS 31+ uses user.ini
        $p = Join-Path $obs $ini
        if (Test-Path $p) {
            $m = Select-String -Path $p -Pattern '^ProfileDir=(.+)$' | Select-Object -First 1
            if ($m) { $profileDir = $m.Matches[0].Groups[1].Value.Trim(); break }
        }
    }
    $candidates = @()
    if ($profileDir) { $candidates += Join-Path $obs "basic\profiles\$profileDir\basic.ini" }
    $candidates += Get-ChildItem (Join-Path $obs "basic\profiles") -Filter basic.ini -Recurse `
        -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | ForEach-Object FullName
    foreach ($basic in $candidates) {
        if (-not (Test-Path $basic)) { continue }
        $text = Get-Content $basic -Raw
        $advanced = $text -match '(?m)^Mode=Advanced'
        $key = if ($advanced) { 'RecFilePath' } else { 'FilePath' }
        $m = [regex]::Match($text, "(?m)^$key=(.+)$")
        if ($m.Success) {
            $dir = $m.Groups[1].Value.Trim() -replace '/', '\'
            if (Test-Path $dir) { return $dir }
        }
    }
    return (Join-Path $env:USERPROFILE "Videos")
}

# --- Pick the recording
if (-not $File) {
    $folder = Get-ObsRecordingFolder
    Say "OBS recordings folder: $folder"
    Start-Sleep -Seconds 3   # give OBS a moment to finish (or remux MKV -> MP4)
    $latest = Get-ChildItem $folder -File | Where-Object { $VideoExt -contains $_.Extension.ToLower() } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latest) { Fail "No recordings found in $folder" }
    # Prefer OBS's remuxed MP4 over the MKV it was made from
    if ($latest.Extension -ieq ".mkv") {
        $mp4 = Join-Path $latest.DirectoryName ($latest.BaseName + ".mp4")
        if (Test-Path $mp4) { $latest = Get-Item $mp4 }
    }
    $age = (Get-Date) - $latest.LastWriteTime
    if ($age.TotalHours -gt 12) {
        $ans = Read-Host "Newest recording is '$($latest.Name)' from $([int]$age.TotalHours) h ago. Use it? (y/n)"
        if ($ans -notmatch '^[yY]') { Fail "Stopped. Pass -File to choose a video." }
    }
    $File = $latest.FullName
}
if (-not (Test-Path $File)) { Fail "File not found: $File" }

# Wait until OBS has stopped writing the file
Say "Recording: $File"
$size = -1
while ($true) {
    $now = (Get-Item $File).Length
    if ($now -eq $size) { break }
    if ($size -ge 0) { Say "  still being written, waiting..." "DarkGray" }
    $size = $now
    Start-Sleep -Seconds 5
}
Say ("Size: {0:N2} GB" -f ($size / 1GB))

# --- Game name (remembered for next time)
New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
$gameFile = Join-Path $StateDir "last_game.txt"
if (-not $Game) {
    $last = if (Test-Path $gameFile) { (Get-Content $gameFile -Raw).Trim() } else { "" }
    $prompt = if ($last) { "Game name [$last]" } else { "Game name (for titles and hashtags)" }
    $Game = (Read-Host $prompt).Trim()
    if (-not $Game) { $Game = $last }
}
if ($Game) { Set-Content -Path $gameFile -Value $Game -Encoding UTF8 }

# --- Make sure the local AI (Ollama) is running
function Test-Ollama {
    try { Invoke-RestMethod "http://localhost:11434/api/tags" -TimeoutSec 3 | Out-Null; return $true }
    catch { return $false }
}
if (-not (Test-Ollama)) {
    Say "Starting Ollama..."
    $ollama = Get-Command ollama -ErrorAction SilentlyContinue
    if ($ollama) { Start-Process $ollama.Source -ArgumentList "serve" -WindowStyle Hidden }
    for ($i = 0; $i -lt 20 -and -not (Test-Ollama); $i++) { Start-Sleep -Seconds 1 }
    if (-not (Test-Ollama)) { Say "Ollama is not running: moments will be ranked without AI judging." "Yellow" }
}

$started = Get-Date

# --- 1. Register the video (prints the project id)
Say "`n[1/3] Registering the recording" "Green"
$ingest = Invoke-Sphura ingest $File 2>&1
$ingest | ForEach-Object { Write-Host "  $_" }
$projectLine = $ingest | Select-String -Pattern '^project\s+([0-9a-f]+)' | Select-Object -First 1
if (-not $projectLine) { Fail "Could not register the video (see above)." }
$ProjectId = $projectLine.Matches[0].Groups[1].Value

# --- 2. Analyse (resumable: re-running continues where it stopped)
Say "`n[2/3] Finding the best moments (about 10-15 min for a 2 h stream)" "Green"
Invoke-Sphura analyze $ProjectId
if ($LASTEXITCODE -ne 0) { Fail "Analysis failed or was stopped. Run this script again to resume." }

# --- 3. Render each style
$name = [IO.Path]::GetFileNameWithoutExtension($File)
$safe = ($name -replace '[\\/:*?"<>|#]', ' ' -replace '\s+', ' ').Trim()
$OutDir = Join-Path $OutRoot ("{0} {1}" -f (Get-Date -Format "yyyy-MM-dd"), $safe)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$i = 0
foreach ($style in $Styles) {
    $i++
    Say "`n[3/3] Rendering '$style' Shorts ($i of $($Styles.Count))" "Green"
    $renderArgs = @("render", $ProjectId, "--style", $style, "--top", "$Top", "--out", (Join-Path $OutDir $style))
    if ($Game) { $renderArgs += @("--game", $Game) }
    Invoke-Sphura @renderArgs
    if ($LASTEXITCODE -ne 0) { Say "  '$style' had problems (see above); continuing" "Yellow" }
}

# --- Done
$count = (Get-ChildItem $OutDir -Recurse -Filter *.mp4 -ErrorAction SilentlyContinue).Count
$mins = [int]((Get-Date) - $started).TotalMinutes
Say "`nDone in $mins min: $count Short(s) in $OutDir" "Green"
Say "Tell SPHURA what you liked:  sphura feedback $ProjectId --good 1 --bad 3" "DarkGray"

try {
    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
    $xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent(
        [Windows.UI.Notifications.ToastTemplateType]::ToastText02)
    $t = $xml.GetElementsByTagName("text")
    [void]$t.Item(0).AppendChild($xml.CreateTextNode("SPHURA: $count Shorts ready"))
    [void]$t.Item(1).AppendChild($xml.CreateTextNode($safe))
    $appId = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($appId).Show(
        [Windows.UI.Notifications.ToastNotification]::new($xml))
} catch { [console]::beep(880, 300) }

Start-Process explorer.exe $OutDir
