@echo off
rem Copies the engine programs built by build-basalt.cmd (src\basalt\bin) into the runnable tree
rem engines\basalt-src\app\bin. The unpatched 0.5.6 build stays in engines\basalt-src\baseline\bin and is never written here.
rem Stop the engine first: Windows does not replace a running program. See BUILD-NOTES-basalt.md.
setlocal
set ROOT=C:\WORKDIR\inkling
set FROM=%ROOT%\src\basalt\bin
set TO=%ROOT%\engines\basalt-src\app\bin
if not exist "%FROM%\basalt.exe" (echo no build in %FROM% & exit /b 1)
if not exist "%TO%" (echo no runnable tree in %TO% & exit /b 1)
for %%F in (basalt.exe basalt-bench.exe basalt-probe.exe) do (
  if exist "%FROM%\%%F" copy /y "%FROM%\%%F" "%TO%\%%F" >nul || (echo cannot write %TO%\%%F - is the engine running? & exit /b 1)
)
echo INSTALLED to %TO%
endlocal
