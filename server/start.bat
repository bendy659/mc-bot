@echo off
rem Сервер для обучения ботов mc-bot (Paper 26.1.2), см. README.md рядом.
rem Остановить — команда stop в этом окне (мир сохранится).
chcp 65001 >nul
cd /d "%~dp0"
java -Xms1G -Xmx3G -jar paper-26.1.2-74.jar --nogui
pause
