@echo off
setlocal

:: Change directory to the location of the batch file
cd /d "%~dp0"

:: Run the application with the '--auto' flag
build\Release\CrackBit.exe --auto

:: Pause to keep the window open after execution
pause
