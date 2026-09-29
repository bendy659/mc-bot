@echo off
rem Старый вариант запуска: Python и Node в двух отдельных консолях.
rem Основной способ теперь - run_ai_bot.bat (одно окно-лаунчер).
setlocal

cd /d "%~dp0"

echo [mc-bot] Starting Python AI controller...
start "mc-bot Python AI" cmd /k "cd /d ""%~dp0"" && npm run ai"

timeout /t 2 /nobreak > nul

echo [mc-bot] Starting Mineflayer bot...
start "mc-bot Mineflayer Bot" cmd /k "cd /d ""%~dp0"" && npm run bot"

echo [mc-bot] Started.
echo [mc-bot] Close the opened console windows to stop the processes.

endlocal
