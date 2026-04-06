# Download Sample Drone Images for ODM Testing
# Run this script AFTER Docker and WebODM are installed

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ODM Sample Data Downloader" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Create directory for samples
$samplesDir = "C:\odm_samples"
if (!(Test-Path $samplesDir)) {
    New-Item -ItemType Directory -Path $samplesDir | Out-Null
    Write-Host "[✓] Created directory: $samplesDir" -ForegroundColor Green
} else {
    Write-Host "[✓] Directory exists: $samplesDir" -ForegroundColor Green
}

Write-Host ""
Write-Host "Downloading sample dataset..." -ForegroundColor Yellow
Write-Host ""

# Download Brighton Beach sample (small dataset, good for testing)
$url = "https://github.com/OpenDroneMap/odm_data/releases/download/0.0.1/brighton_beach.zip"
$output = "$samplesDir\brighton_beach.zip"

try {
    Write-Host "Downloading from: $url" -ForegroundColor Cyan
    Write-Host "This may take a few minutes (dataset is ~50-100MB)..." -ForegroundColor Yellow
    
    # Download with progress
    $webClient = New-Object System.Net.WebClient
    $webClient.DownloadFile($url, $output)
    
    Write-Host "[✓] Download complete!" -ForegroundColor Green
    Write-Host ""
    
    # Extract
    Write-Host "Extracting files..." -ForegroundColor Yellow
    Expand-Archive -Path $output -DestinationPath "$samplesDir\brighton_beach" -Force
    Write-Host "[✓] Extraction complete!" -ForegroundColor Green
    Write-Host ""
    
    # Clean up zip
    Remove-Item $output
    
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Sample Data Ready!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Location: $samplesDir\brighton_beach" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Yellow
    Write-Host "1. Make sure Docker Desktop is running"
    Write-Host "2. Start WebODM: cd C:\Users\z7aa\WebODM; .\webodm.bat start"
    Write-Host "3. Open browser: http://localhost:8000"
    Write-Host "4. Upload images from: $samplesDir\brighton_beach\images"
    Write-Host ""
    
} catch {
    Write-Host "[✗] Download failed: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Manual download option:" -ForegroundColor Yellow
    Write-Host "1. Go to: https://github.com/OpenDroneMap/odm_data/releases"
    Write-Host "2. Download brighton_beach.zip"
    Write-Host "3. Extract to: $samplesDir"
    Write-Host ""
}

Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
