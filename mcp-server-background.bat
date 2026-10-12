@echo off
REM ---------------------------------------------------------------------------
REM Background launcher for mcp-server.bat, for a box reached over SSH.
REM
REM A process started inside an ssh session is killed when the session ends, so
REM this is meant to run as an interactive scheduled task in the logged-on
REM desktop session (wangp-cli's windows/windows.sh creates and runs it):
REM
REM   schtasks /create /f /tn WanGP-MCP /sc once /st 00:00 /it /tr "<this file>"
REM   schtasks /run /tn WanGP-MCP
REM
REM Differences from running mcp-server.bat directly:
REM   - binds 127.0.0.1, so the server is reachable ONLY through an ssh tunnel
REM     (ssh -L 17867:127.0.0.1:7866 ...) -- port 7866 never faces the network.
REM   - output goes to mcp-server.log (overwritten each start; *.log is ignored)
REM     instead of a console window, so it can be read over ssh.
REM   - per-launch knobs come from mcp-server.env.bat when present (git-ignored;
REM     windows.sh writes it), e.g.
REM       set "VRAM_SAFETY=0.5"
REM       set "MCP_HOST=0.0.0.0"
REM ---------------------------------------------------------------------------
cd /d "%~dp0"
set "MCP_HOST=127.0.0.1"
if exist mcp-server.env.bat call mcp-server.env.bat
call mcp-server.bat > mcp-server.log 2>&1
