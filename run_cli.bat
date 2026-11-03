@echo off
chcp 65001 >nul
echo ========================================================
echo  KHOI DONG CHATBOT AI V4.0 (TERMINAL CLI)
echo ========================================================
py chatbot.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [LOI] Khong the chay bang lenh 'py', dang thu tim python.exe truc tiep...
    "C:\Users\DUC\AppData\Local\Python\pythoncore-3.14-64\python.exe" chatbot.py
)
pause

