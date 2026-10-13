# Check download status for Kavach models

Write-Host "`n" -NoNewline
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  KAVACH MODEL DOWNLOAD STATUS" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan

# Check for completed downloads
$completed = Get-ChildItem "models" -Filter "*.gguf" -Recurse -ErrorAction SilentlyContinue | 
    Where-Object { $_.Directory.Name -ne "download" }

if ($completed) {
    Write-Host "`n✅ COMPLETED DOWNLOADS:" -ForegroundColor Green
    $completed | ForEach-Object {
        $sizeMB = [math]::Round($_.Length/1MB,2)
        $sizeGB = [math]::Round($_.Length/1GB,3)
        Write-Host "   ✓ $($_.Name)" -ForegroundColor Green
        Write-Host "     Size: $sizeGB GB ($sizeMB MB)" -ForegroundColor Gray
    }
} else {
    Write-Host "`n⏳ No completed downloads yet" -ForegroundColor Yellow
}

# Check for downloads in progress
$inProgress = Get-ChildItem "models\.cache" -Recurse -ErrorAction SilentlyContinue | 
    Where-Object { $_.Name -like "*.lock" }

if ($inProgress) {
    Write-Host "`n📥 DOWNLOADS IN PROGRESS:" -ForegroundColor Cyan
    $inProgress | ForEach-Object {
        $modelName = $_.Name -replace ".lock", ""
        Write-Host "   ⏳ $modelName" -ForegroundColor Yellow
        
        # Try to find partial file
        $partial = Get-Item ($_.FullName -replace ".lock", "") -ErrorAction SilentlyContinue
        if ($partial) {
            $sizeMB = [math]::Round($partial.Length/1MB,2)
            Write-Host "      Downloaded so far: $sizeMB MB" -ForegroundColor Gray
        }
    }
} else {
    Write-Host "`n⏸️  No active downloads" -ForegroundColor Gray
}

# Expected models
$expected = @(
    @{Name="qwen2.5-1.5b-instruct-q4_k_m.gguf"; Size="1.3 GB"},
    @{Name="bge-m3-q8_0.gguf"; Size="2.0 GB"},
    @{Name="qwen2.5-7b-instruct-q4_k_m.gguf"; Size="5.0 GB"},
    @{Name="qwen2.5-coder-7b-instruct-q4_k_m.gguf"; Size="4.6 GB"},
    @{Name="qwen2-vl-7b-instruct-q4_k_m.gguf"; Size="6.1 GB"}
)

Write-Host "`n📋 EXPECTED MODELS (Total: ~19 GB):" -ForegroundColor White
$expected | ForEach-Object {
    $exists = Test-Path "models\$($_.Name)"
    $status = if ($exists) { "✓" } else { "⏳" }
    $color = if ($exists) { "Green" } else { "Gray" }
    Write-Host "   $status $($_.Name) - $($_.Size)" -ForegroundColor $color
}

Write-Host "`n" -NoNewline
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
