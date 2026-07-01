@echo off
setlocal EnableExtensions

set "DATA_ROOT=D:\Workspace\large-4dstem-analysis\data\0617-4d"
set "TILE_NAME=phase_map_real_ti_only_best_candidate_clean.png"
set "OUTPUT_BASE=%DATA_ROOT%\phase_map_real_ti_only_best_candidate_clean_merged"

if not "%~1"=="" set "OUTPUT_BASE=%~dpn1"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'Stop';" ^
  "$dataRoot = '%DATA_ROOT%';" ^
  "$tileName = '%TILE_NAME%';" ^
  "$outputBase = '%OUTPUT_BASE%';" ^
  "Add-Type -AssemblyName System.Drawing;" ^
  "$tiles = @();" ^
  "Get-ChildItem -LiteralPath $dataRoot -Directory | ForEach-Object {" ^
  "  if ($_.Name -match '^1_(\d+)_(\d+)_(\d+)_(\d+)$') {" ^
  "    $a0 = [int]$Matches[1]; $a1 = [int]$Matches[2]; $b0 = [int]$Matches[3]; $b1 = [int]$Matches[4];" ^
  "    if (($a1 - $a0) -eq 64 -and ($b1 - $b0) -eq 64 -and $a0 -ge 0 -and $a1 -le 512 -and $b0 -ge 0 -and $b1 -le 512) {" ^
  "      $path = Join-Path $_.FullName $tileName;" ^
  "      if (Test-Path -LiteralPath $path) { $tiles += [pscustomobject]@{ Path = $path; A = $a0; B = $b0; Name = $_.Name } }" ^
  "    }" ^
  "  }" ^
  "};" ^
  "if ($tiles.Count -ne 64) {" ^
  "  $found = ($tiles | Sort-Object A, B | ForEach-Object { $_.Name }) -join ', ';" ^
  "  throw ('Expected 64 tiles, found {0}. Found: {1}' -f $tiles.Count, $found);" ^
  "}" ^
  "$first = [System.Drawing.Image]::FromFile($tiles[0].Path);" ^
  "$tileWidth = $first.Width; $tileHeight = $first.Height; $first.Dispose();" ^
  "$loaded = @();" ^
  "try {" ^
  "  foreach ($tile in $tiles) {" ^
  "    $img = [System.Drawing.Image]::FromFile($tile.Path);" ^
  "    if ($img.Width -ne $tileWidth -or $img.Height -ne $tileHeight) { throw ('Tile size mismatch in {0}: {1}x{2}, expected {3}x{4}' -f $tile.Path, $img.Width, $img.Height, $tileWidth, $tileHeight); }" ^
  "    $loaded += [pscustomobject]@{ Image = $img; A = $tile.A; B = $tile.B; Path = $tile.Path };" ^
  "  }" ^
  "  function Save-Merged($rowField, $colField, $suffix) {" ^
  "    $cols = $loaded | ForEach-Object { $_.$colField } | Select-Object -Unique | Sort-Object;" ^
  "    $rows = $loaded | ForEach-Object { $_.$rowField } | Select-Object -Unique | Sort-Object -Descending;" ^
  "    if ($cols.Count -ne 8 -or $rows.Count -ne 8) { throw ('Expected an 8x8 grid for {0}, found {1} columns and {2} rows.' -f $suffix, $cols.Count, $rows.Count); }" ^
  "    $canvas = New-Object System.Drawing.Bitmap ($tileWidth * 8), ($tileHeight * 8);" ^
  "    $graphics = [System.Drawing.Graphics]::FromImage($canvas);" ^
  "    try {" ^
  "      $graphics.Clear([System.Drawing.Color]::White);" ^
  "      foreach ($item in $loaded) {" ^
  "        $col = [array]::IndexOf($cols, $item.$colField);" ^
  "        $row = [array]::IndexOf($rows, $item.$rowField);" ^
  "        $graphics.DrawImage($item.Image, ($col * $tileWidth), ($row * $tileHeight), $tileWidth, $tileHeight);" ^
  "      }" ^
  "      $outputFile = $outputBase + '_' + $suffix + '.png';" ^
  "      $outDir = Split-Path -Parent $outputFile;" ^
  "      if ($outDir) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null; }" ^
  "      $canvas.Save($outputFile, [System.Drawing.Imaging.ImageFormat]::Png);" ^
  "      Write-Host ('[done] Saved {0}: {1}' -f $suffix, $outputFile);" ^
  "    } finally {" ^
  "      if ($graphics) { $graphics.Dispose(); }" ^
  "      if ($canvas) { $canvas.Dispose(); }" ^
  "    }" ^
  "  }" ^
  "  Save-Merged 'A' 'B' 'y_rows_x_cols';" ^
  "  Save-Merged 'B' 'A' 'xy_swapped';" ^
  "} finally {" ^
  "  foreach ($item in $loaded) { if ($item.Image) { $item.Image.Dispose(); } }" ^
  "}"

if errorlevel 1 (
    echo [error] Failed to merge orientation phase maps.
    exit /b %ERRORLEVEL%
)

exit /b 0
