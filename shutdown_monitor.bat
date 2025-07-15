
@echo off
REM shutdown_monitor.bat - Monitors for shutdown signal and kills processes

:loop
REM Check if shutdown signal file exists
if exist "SHUTDOWN_SIGNAL" (
    echo Shutdown signal detected! Killing all LiangLocal processes...
    
    REM Kill processes on ports 8000 and 8001
    FOR /F "tokens=5" %%P IN ('netstat -aon ^| findstr ":8000.*LISTENING"') DO (
        IF "%%P" NEQ "0" (
            echo Killing process %%P on port 8000...
            taskkill /PID %%P /F /T > nul 2>&1
        )
    )
    
    FOR /F "tokens=5" %%P IN ('netstat -aon ^| findstr ":8001.*LISTENING"') DO (
        IF "%%P" NEQ "0" (
            echo Killing process %%P on port 8001...
            taskkill /PID %%P /F /T > nul 2>&1
        )
    )
    
    REM Also kill any python processes running main.py or launch.py
    taskkill /IM python.exe /FI "WINDOWTITLE eq *launch.py*" /F /T > nul 2>&1
    taskkill /IM python.exe /FI "WINDOWTITLE eq *main.py*" /F /T > nul 2>&1
    
    REM Kill npm/node processes
    taskkill /IM node.exe /F /T > nul 2>&1
    taskkill /IM npm.cmd /F /T > nul 2>&1
    
    REM Clean up the shutdown signal file
    del "SHUTDOWN_SIGNAL" > nul 2>&1
    
    echo All processes killed. Monitor shutting down.
    timeout /t 2 /nobreak > nul
    exit
)

REM Wait 1 second before checking again
timeout /t 1 /nobreak > nul
goto loop