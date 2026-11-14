@echo off
rem Starts the VLM Prompter GUI in its own window and opens it in the browser (http://127.0.0.1:8790).
rem Close that window (or Ctrl+C in it) to stop the server.
cd /d "%~dp0"
start "VLM Prompter" "%~dp0.venv\Scripts\python.exe" -m vlm_prompter serve
