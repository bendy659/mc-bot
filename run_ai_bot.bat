@echo off
rem Запуск mc-bot одним окном: py\launcher.py сам поднимает Python и Node
rem и собирает их логи в одно место (старый вариант с двумя консолями -
rem run_ai_bot_consoles.bat).
cd /d "%~dp0"

where pythonw >nul 2>nul
if %errorlevel%==0 (
    start "" pythonw py\launcher.py
) else (
    python py\launcher.py
)
