@echo off
REM ---------------------------------------------------------------------------
REM WanGP MCP server launcher (+ outputs file server)
REM
REM Starts two things, both bound to all interfaces (0.0.0.0):
REM   1. WanGP MCP server (Streamable HTTP) on MCP_PORT
REM        -> MCP client connects to:  http://%MCP_HOST%:%MCP_PORT%/mcp
REM   2. An HTTP server for the WanGP outputs folder on HTTP_PORT, which both
REM      serves generated media AND accepts uploads (for reference images).
REM        -> download generated media : http://%MCP_HOST%:%HTTP_PORT%/<filename>
REM        -> upload a reference image : POST http://%MCP_HOST%:%HTTP_PORT%/upload
REM           (multipart field "files", no auth, lands in outputs\)
REM
REM HTTP_PORT reuses the port web-ui.bat normally uses (Gradio default 7860),
REM which is free in MCP mode since the web UI is not launched -- so no new
REM firewall rule is needed beyond the one already covering that port.
REM
REM NOTE: No endpoint has authentication, including file UPLOADS -- anyone who
REM can reach these ports can read outputs and write files into outputs\.
REM Binding 0.0.0.0 listens on ALL interfaces, so exposure is limited ONLY by
REM your firewall scope -- make sure these ports are firewalled to the VPN/LAN.
REM
REM Two console windows open: this one (MCP server) and "WanGP Outputs HTTP".
REM Close BOTH windows to fully stop the servers.
REM ---------------------------------------------------------------------------

REM Bind all interfaces (like web-ui.bat's --listen) so it's reachable on the LAN/VPN.
set MCP_HOST=0.0.0.0
set MCP_PORT=7866
set HTTP_PORT=7860

call venv\Scripts\activate.bat

REM Make sure the outputs folder exists before serving it
if not exist outputs mkdir outputs

REM Start the outputs file server (download + unauthenticated upload) in its own window.
REM --allow-replace keeps the uploaded filename intact (so the agent can predict the path).
start "WanGP Outputs HTTP" venv\Scripts\python.exe -m uploadserver %HTTP_PORT% --bind %MCP_HOST% --directory outputs --allow-replace

REM Run the MCP server in the foreground (this window)
python wgp.py --mcp --mcp-transport streamable-http --mcp-host %MCP_HOST% --mcp-port %MCP_PORT% --mcp-console-output
