@echo off
chcp 65001 >nul
echo ========================================================
echo  KHOI DONG WEB SERVER CHATBOT AI V4.0
echo  Dia chi: http://127.0.0.1:5000
echo ========================================================
start "" http://127.0.0.1:5000
py app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [LOI] Khong the chay bang lenh 'py', dang thu tim python.exe truc tiep...
    "C:\Users\DUC\AppData\Local\Python\pythoncore-3.14-64\python.exe" app.py
)
pause

