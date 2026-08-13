@echo off
REM ---------------------------------------------------------------------------
REM WanGP MCP server launcher
REM
REM ONE process serves everything, bound to all interfaces (0.0.0.0):
REM   - MCP (Streamable HTTP):       http://%MCP_HOST%:%MCP_PORT%/mcp
REM   - download generated media:    GET  http://%MCP_HOST%:%MCP_PORT%/files/<relpath>
REM   - list a directory:            GET  http://%MCP_HOST%:%MCP_PORT%/files/  (plain text)
REM   - upload a reference image:    POST http://%MCP_HOST%:%MCP_PORT%/files/upload
REM       (multipart field "files", no auth, lands in outputs\)
REM
REM The file routes are served by the MCP server itself (fork-only,
REM shared/mcp_files.py) -- the separate `uploadserver` process on port 7860
REM that this script used to start is gone. Port 7860 stays free for
REM web-ui.bat's Gradio UI.
REM
REM NOTE: No endpoint has authentication, including file UPLOADS -- anyone who
REM can reach this port can read outputs and write files into outputs\.
REM Binding 0.0.0.0 listens on ALL interfaces, so exposure is limited ONLY by
REM your firewall scope -- make sure the port is firewalled to the VPN/LAN.
REM ---------------------------------------------------------------------------

REM Bind all interfaces (like web-ui.bat's --listen) so it's reachable on the LAN/VPN.
set MCP_HOST=0.0.0.0
set MCP_PORT=7866

REM ---------------------------------------------------------------------------
REM Memory tuning (this box: RTX 5080 16 GB VRAM, 64 GB system RAM)
REM
REM Big models (MiniMax H3 20B/33B) failed to generate on the stock defaults.
REM WanGP reports that as "unsufficient RAM ... perc_reserved_mem_max", but the
REM message is a catch-all: the real traceback was
REM   torch.AcceleratorError: CUDA error: out of memory
REM while VAE-encoding a reference image. So the knob that actually matters here
REM is VRAM_SAFETY, not PERC_RESERVED.
REM
REM   VRAM_SAFETY    Fraction of VRAM mmgp may fill with preloaded model weights
REM                  (wgp.py default 0.8 -> up to ~12.8 GB of a 16 GB card, which
REM                  leaves too little for activations). Lower = more headroom.
REM                  If a model still OOMs, step down: 0.5 -> 0.4 -> 0.3.
REM
REM   PERC_RESERVED  Fraction of system RAM pinned for fast RAM->VRAM transfers.
REM                  Default 0 = auto, which topped out near 25 GB and left the
REM                  VAE / vision + video encoders unpinned ("no reserved RAM
REM                  left. Transfer speed ... may be slower"). SPEED ONLY -- it
REM                  cannot fix a CUDA OOM. Must stay BELOW 0.5.
REM
REM   PROFILE        mmgp memory profile 1-5 (wgp.py default 4 = LowRAM_LowVRAM).
REM                  Try 5 (VerylowRAM_LowVRAM, offloads hardest) if lowering
REM                  VRAM_SAFETY alone is not enough. With 64 GB of RAM, 2
REM                  (HighRAM_LowVRAM) may instead be faster once it fits.
REM
REM Change one at a time so it stays clear which one moved the needle.
REM
REM Each can be overridden for a single launch without editing this file.
REM Quote the assignment -- `set VAR=0.35 && ...` captures the space before the
REM `&&` into the value:
REM   set "VRAM_SAFETY=0.35" && mcp-server.bat
REM ...or just set it on its own line first, then run the script.
REM
REM Known-good values on this box:
REM   Pruned 20B (fl2va/ref2va_pruned) ... VRAM_SAFETY=0.5  works
REM   Full 33B   (fl2va/ref2va)        ... needs LOWER; 0.5 OOMs in the VAE
REM                                        encode. Try 0.35, then 0.3, then
REM                                        PROFILE=5.
REM ---------------------------------------------------------------------------
if "%VRAM_SAFETY%"==""   set VRAM_SAFETY=0.5
if "%PERC_RESERVED%"=="" set PERC_RESERVED=0.45
if "%PROFILE%"==""       set PROFILE=4

call venv\Scripts\activate.bat

REM Make sure the outputs folder exists before serving it
if not exist outputs mkdir outputs

REM Run the MCP server (which also serves /files/*) in the foreground
python wgp.py --mcp --mcp-transport streamable-http --mcp-host %MCP_HOST% --mcp-port %MCP_PORT% --mcp-console-output --profile %PROFILE% --vram-safety-coefficient %VRAM_SAFETY% --perc-reserved-mem-max %PERC_RESERVED%
