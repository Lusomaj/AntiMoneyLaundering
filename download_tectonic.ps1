$dest = "E:\MASTERSProject\AMLProject\tools"
if (!(Test-Path $dest)) {
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
}
$zipPath = Join-Path $dest "tectonic.zip"
$exePath = Join-Path $dest "tectonic.exe"

if (!(Test-Path $exePath)) {
    Write-Host "Downloading Tectonic compiler..."
    $url = "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic@0.17.0/tectonic-0.17.0-x86_64-pc-windows-msvc.zip"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $url -OutFile $zipPath
    Write-Host "Extracting Tectonic..."
    Expand-Archive -Path $zipPath -DestinationPath $dest -Force
    Remove-Item $zipPath -Force
}

if (Test-Path $exePath) {
    Write-Host "Tectonic is ready:"
    & $exePath --version
} else {
    Write-Host "Failed to extract tectonic.exe"
}
