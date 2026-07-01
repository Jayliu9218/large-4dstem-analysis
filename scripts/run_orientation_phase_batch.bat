@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "DATA_ROOT=D:\Workspace\large-4dstem-analysis\data\0617-4d"
set "SCRIPT=orientation_phase_multi_axis_v6_optimized_ti_with_ws2_qc.py"
set "DONE_MARKER=phase_summary_v6_optimized.json"

cd /d "%~dp0"
if errorlevel 1 (
    echo [error] Failed to switch to repository root: %~dp0
    exit /b 1
)

if not exist "%SCRIPT%" (
    echo [error] Missing script: %SCRIPT%
    exit /b 1
)

call conda activate large-4dstem
if errorlevel 1 (
    echo [error] Failed to activate Conda environment: large-4dstem
    exit /b %ERRORLEVEL%
)

for /L %%Y in (0,64,448) do (
    set /A Y1=%%Y+64
    for /L %%X in (0,64,448) do (
        set /A X1=%%X+64
        set "FILENAME=1_%%Y_!Y1!_%%X_!X1!.h5"
        set "STEM=1_%%Y_!Y1!_%%X_!X1!"
        set "INPUT_FILE=%DATA_ROOT%\!FILENAME!"
        set "SUMMARY_FILE=%DATA_ROOT%\!STEM!\%DONE_MARKER%"

        if exist "!SUMMARY_FILE!" (
            echo [skip] !FILENAME! already completed: !SUMMARY_FILE!
        ) else (
            if not exist "!INPUT_FILE!" (
                echo [error] Missing input file: !INPUT_FILE!
                exit /b 2
            )

            echo [run] !FILENAME!
            python "%SCRIPT%" --data-file "!FILENAME!"
            if errorlevel 1 (
                set "EXIT_CODE=!ERRORLEVEL!"
                echo [error] Failed while processing !FILENAME! with exit code !EXIT_CODE!
                exit /b !EXIT_CODE!
            )
        )
    )
)

echo [done] Processed orientation phase batch.
exit /b 0
