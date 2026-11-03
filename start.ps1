param([switch]$Build)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (!(Test-Path $python)) { throw 'Create .venv and install src-backend/requirements.txt first. See README.md.' }
$nodeCommand = Get-Command node -ErrorAction SilentlyContinue
if ($nodeCommand) { $node = $nodeCommand.Source } else {
    $node = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
}
if (!(Test-Path $node)) { throw 'Install Node.js and npm first. See README.md.' }
$env:PATH = (Split-Path $node) + ';' + $env:PATH
$frontend = Join-Path $PSScriptRoot 'src-frontend'
if (!(Test-Path "$frontend/node_modules/react-scripts/bin/react-scripts.js")) { throw 'Install frontend dependencies with npm ci first.' }
New-Item -ItemType Directory -Force .logs | Out-Null
if ($Build) {
    Push-Location $frontend
    try { & $node node_modules/react-scripts/bin/react-scripts.js build; if ($LASTEXITCODE) { throw 'Frontend build failed' } } finally { Pop-Location }
    exit
}
$backend = $null
try {
    $backend = Start-Process -FilePath $python -ArgumentList '-u', 'src-backend/run.py' -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput "$PSScriptRoot/.logs/backend.log" -RedirectStandardError "$PSScriptRoot/.logs/backend-error.log"
    Write-Host 'Backend: http://localhost:5000/health | Frontend: http://localhost:3000'
    Write-Host 'Press Ctrl+C to stop. Backend logs: .logs/'
    $env:BROWSER = 'none'
    Push-Location $frontend
    try { & $node node_modules/react-scripts/bin/react-scripts.js start } finally { Pop-Location }
} finally {
    if ($backend -and !$backend.HasExited) { Stop-Process -Id $backend.Id }
}
