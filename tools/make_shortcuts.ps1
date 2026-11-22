# Creates the desktop shortcuts of the Inkling launch profiles (and a copy of each in inkling\shortcuts).
#   powershell -ExecutionPolicy Bypass -File make_shortcuts.ps1
# The existing shortcut "Strata + omp-web.lnk" is not touched. Run again after moving the folder.
$ErrorActionPreference = 'Stop'
$inkling = Split-Path -Parent $PSScriptRoot
$pythonw = 'D:\AI\models\Strata\control\python\pythonw.exe'
$desktop = [Environment]::GetFolderPath('Desktop')
$copies  = Join-Path $inkling 'shortcuts'
New-Item -ItemType Directory -Force -Path $copies | Out-Null
$chip    = "$env:SystemRoot\System32\shell32.dll,12"
$strata  = 'D:\AI\models\Strata\strata-desktop.exe,0'
$items = @(
    @{ Name = 'Inkling баланс 262k + omp-web';   Args = "`"$inkling\start_inkling.pyw`" balanced --web"; Icon = $chip;
       Text = 'Inkling 2 Mini, окно 262144 (NInfer, KV rk8v4) и веб-окно агента' },
    @{ Name = 'Inkling макс 512k + omp-web';     Args = "`"$inkling\start_inkling.pyw`" max --web";      Icon = $chip;
       Text = 'Inkling 2 Mini, окно 524288 (NInfer, KV rk4v4) и веб-окно агента' },
    @{ Name = 'Inkling скорость 131k + omp-web'; Args = "`"$inkling\start_inkling.pyw`" speed --web";    Icon = $chip;
       Text = 'Inkling 2 Mini, окно 131072 (NInfer, точный KV) и веб-окно агента' },
    @{ Name = 'Вернуть Qwen + omp-web';          Args = "`"$inkling\start_qwen.pyw`" --web";             Icon = $strata;
       Text = 'Выгрузить Inkling, загрузить Qwen3.8 Flash Next в Strata и открыть веб-окно агента' },
    @{ Name = 'Inkling выгрузить';               Args = "`"$inkling\start_qwen.pyw`" --stop-only";       Icon = $chip;
       Text = 'Выгрузить Inkling и вернуть агентам модель Qwen по умолчанию (Strata не запускается)' }
)
$shell = New-Object -ComObject WScript.Shell
foreach ($item in $items) {
    foreach ($dir in @($desktop, $copies)) {
        $link = $shell.CreateShortcut((Join-Path $dir ($item.Name + '.lnk')))
        $link.TargetPath       = $pythonw
        $link.Arguments        = $item.Args
        $link.WorkingDirectory = $inkling
        $link.IconLocation     = $item.Icon
        $link.Description      = $item.Text
        $link.Save()
    }
    Write-Output ("created: " + $item.Name)
}
