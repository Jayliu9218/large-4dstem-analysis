@echo off
setlocal EnableExtensions EnableDelayedExpansion

pushd "%~dp0\.."
if errorlevel 1 (
    echo Failed to enter repository root.
    exit /b 1
)

call conda activate large-4dstem
if errorlevel 1 (
    echo Failed to activate conda environment: large-4dstem
    popd
    exit /b 1
)

set "INPUT=data\0617-4d\1_512x512_ss15.63nm_0.55ms_c2 50um_CL91mm_0.75mrad_spot7_0.022nA_GL3_mag12500k_12b 0913.mib"
set "OUTPUT_DIR=data\0617-4d"

for /L %%Y in (0,64,448) do (
    set /A Y1=%%Y + 64
    for /L %%X in (0,64,448) do (
        set /A X1=%%X + 64
        set "OUTPUT=!OUTPUT_DIR!\1_%%Y_!Y1!_%%X_!X1!.h5"

        if exist "!OUTPUT!" (
            echo Skipping existing tile: !OUTPUT!
        ) else (
            echo Exporting tile: --nav-crop %%Y !Y1! %%X !X1! --output "!OUTPUT!"
            python -m fourdstem_pipeline.cli crop-export ^
                --input "!INPUT!" ^
                --output "!OUTPUT!" ^
                --nav-crop %%Y !Y1! %%X !X1! ^
                --mem MEMMAP ^
                --scan-shape 512 512
            if errorlevel 1 (
                set "ERR=!ERRORLEVEL!"
                echo Crop export failed for tile: %%Y !Y1! %%X !X1!
                popd
                exit /b !ERR!
            )
        )
    )
)

echo All 64x64 crop exports completed.
popd
exit /b 0
