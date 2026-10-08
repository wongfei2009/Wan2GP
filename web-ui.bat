@echo off
if exist .venv\Scripts\activate.bat (call .venv\Scripts\activate.bat) else (call venv\Scripts\activate.bat)
python wgp.py --listen