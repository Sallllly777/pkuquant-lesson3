param(
    [ValidateSet('serve','demo','audit','test')][string]$Command = 'serve',
    [int]$Symbol = 600036,
    [int]$Port = 8765,
    [switch]$All
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$candidates = @(
    $env:L2SIM_PYTHON,
    "$env:USERPROFILE\AppData\Local\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:USERPROFILE\anaconda3\python.exe",
    'python'
) | Where-Object { $_ }
$runtime = $null
foreach ($candidate in $candidates) {
    try {
        & $candidate -B -c 'import pyarrow' 2>$null
        if ($LASTEXITCODE -eq 0) { $runtime = $candidate; break }
    } catch { }
}
if (-not $runtime) { throw 'Python with pyarrow is required. Install requirements.txt into your chosen environment and set L2SIM_PYTHON.' }
if ($Command -eq 'test') {
    & $runtime -B -m unittest discover -s tests -v
} else {
    $arguments = @('-B','-m','l2sim',$Command,'--symbol',"$Symbol",'--port',"$Port")
    if ($All) { $arguments += '--all' }
    & $runtime @arguments
}
exit $LASTEXITCODE
