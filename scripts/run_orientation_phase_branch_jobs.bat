@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem External branch-job launcher for orientation_phase_multi_axis_v6_optimized_ti_with_ws2_qc.py.
rem Usage:
rem   run_orientation_phase_branch_jobs.bat 1_0_64_0_64.h5 [coarse|fine] [run-control|skip-control]

set "SCRIPT=orientation_phase_multi_axis_v6_optimized_ti_with_ws2_qc.py"
set "PYTHON=C:\Users\jayliu\.conda\envs\large-4dstem\python.exe"
set "DATA_FILE=%~1"
set "MODE=%~2"
set "CONTROL=%~3"

if "%DATA_FILE%"=="" (
    echo [error] Missing data filename, e.g. 1_0_64_0_64.h5
    exit /b 2
)
if "%MODE%"=="" set "MODE=coarse"
if "%CONTROL%"=="" set "CONTROL=run-control"

cd /d "%~dp0"
if errorlevel 1 exit /b %ERRORLEVEL%

if not exist "%SCRIPT%" (
    echo [error] Missing script: %SCRIPT%
    exit /b 2
)
if not exist "%PYTHON%" (
    echo [error] Missing Python: %PYTHON%
    exit /b 2
)

set "COMMON=--data-file %DATA_FILE% --mode %MODE% --output-tag branch_jobs_%MODE% --%CONTROL%"

call :run_branch Ti-bcc 0,1,1
call :run_branch Ti-bcc 0,0,1
call :run_branch Ti-bcc 1,1,1
call :run_branch Ti-hcp 1,0,0
call :run_branch Ti-hcp 0,0,1
call :run_branch Ti-hcp 1,1,0

if /I "%CONTROL%"=="run-control" (
    call :run_branch WS2-control 0,0,1
)

echo [aggregate] %DATA_FILE%
"%PYTHON%" "%SCRIPT%" --data-file "%DATA_FILE%" --mode "%MODE%" --output-tag "branch_jobs_%MODE%_aggregated" --aggregate-branches "..\data\0617-4d\%~n1\branch_jobs_%MODE%"
if errorlevel 1 exit /b %ERRORLEVEL%

echo [done] Branch jobs aggregated.
exit /b 0

:run_branch
set "PHASE=%~1"
set "AXIS=%~2"
echo [branch] !PHASE! !AXIS!
"%PYTHON%" "%SCRIPT%" %COMMON% --branch-only --phase "!PHASE!" --fiber-axis "!AXIS!"
if errorlevel 1 exit /b %ERRORLEVEL%
exit /b 0
