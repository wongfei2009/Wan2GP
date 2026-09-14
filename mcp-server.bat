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
REM --mcp-api-version 1 is REQUIRED for the `wangp` CLI, as of WanGP 13.
REM Upstream made API v2 the default: build_server_for_session's api_tool()
REM registers NOTHING under v2 and shared/mcp_v2.py registers a different,
REM compact tool surface instead -- so wangp_generate / wangp_get_job /
REM wangp_list_models simply vanish and every CLI call fails as an unknown
REM tool. The fork-only tools stay registered either way (they use @mcp.tool()
REM directly), which makes the breakage look partial rather than total.
REM
REM --mcp-allow-read-file-system is REQUIRED for the `wangp` CLI, as of WanGP
REM 12.60. Upstream now routes every generation through _resolve_generation_media
REM and rejects filesystem paths by default, for every attachment key the CLI
REM sends (image_start/end, image_refs, image_guide/mask, video_guide/guide2/
REM mask/source, audio_guide/guide2/source, ...) -- without the flag each one
REM fails with "Direct filesystem paths are disabled". The sanctioned
REM alternatives (Gallery ids, the wangp_create_gallery_upload PUT-URL flow) are
REM not what the CLI speaks. Paths arrive relative (outputs\<name>) and resolve
REM against THIS process's CWD, i.e. the repo root -- so run the .bat from here.
REM The flag also enables wangp_list_files / wangp_query_file, i.e. arbitrary
REM server-file reads, so it widens the same no-auth exposure as the note below.
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
REM                  (0.8 -> ~12.8 GB of a 16 GB card; 0.5 -> ~8 GB). It is a
REM                  TRADE, not a safety dial: higher keeps more weights resident
REM                  (faster, and essential when one component must not stream),
REM                  lower leaves more room for activations (survives big VAE
REM                  encodes). Video wants it LOW, audio wants it HIGH -- see the
REM                  per-model box below. If a model OOMs, step down:
REM                  0.8 -> 0.5 -> 0.35 -> 0.3.
REM
REM   PERC_RESERVED  Fraction of system RAM pinned for fast RAM->VRAM transfers.
REM                  Default 0 = auto, which topped out near 25 GB and left the
REM                  VAE / vision + video encoders unpinned ("no reserved RAM
REM                  left. Transfer speed ... may be slower"). SPEED ONLY -- it
REM                  cannot fix a CUDA OOM. Must stay BELOW 0.5.
REM
REM   PROFILE        mmgp memory profile 1-5. DELIBERATELY UNSET BY DEFAULT --
REM                  setting it is NOT free, because --profile is a global
REM                  FORCE, not a default. WanGP already keeps a separate
REM                  memory profile PER OUTPUT TYPE (server_config's
REM                  video_profile / image_profile / audio_profile, the last
REM                  defaulting to 3.5), and wgp.py:3388-3390 collapses all
REM                  three onto --profile whenever it is passed.
REM
REM                  That collapse costs real speed on AUDIO. The cg/vllm LM
REM                  decoder engines require a profile that keeps the model
REM                  wholly in VRAM -- wgp.py:4142 tests int(profile) in [1,3],
REM                  which 3.5 passes and 4 does not -- so forcing 4 silently
REM                  dropped every autoregressive audio model (YuE2, ACE-Step,
REM                  MiniMax Music, Kugel) onto the legacy decoder:
REM                    "Unable to use LM Engine 'vllm' as it requires a Memory
REM                     Profile such as 1,3 or 3+ ... Switching to Legacy"
REM                    "[YuE2] AR LM engine: legacy (CUDA graphs: off; Triton
REM                     decoder kernels: off; attention: PyTorch SDPA)"
REM                  Upstream says the same in docs/CHANGELOG.md for Kugel
REM                  Audio: 16GB VRAM + profile 1/3/3+, "or you will have to go
REM                  the slow way with other Profiles".
REM
REM                  So leave it unset and let the server pick per type. Set it
REM                  only to force one for a whole launch: 5
REM                  (VerylowRAM_LowVRAM, offloads hardest) if lowering
REM                  VRAM_SAFETY alone is not enough for a big VIDEO model, or
REM                  2 (HighRAM_LowVRAM) which may be faster once it fits with
REM                  64 GB of RAM. Doing so re-flattens audio back onto the
REM                  legacy decoder, so prefer per-launch over permanent.
REM
REM Change one at a time so it stays clear which one moved the needle.
REM
REM Each can be overridden for a single launch without editing this file.
REM Quote the assignment -- `set VAR=0.35 && ...` captures the space before the
REM `&&` into the value:
REM   set "VRAM_SAFETY=0.35" && mcp-server.bat
REM ...or just set it on its own line first, then run the script.
REM
REM ###########################################################################
REM #  DEFAULT IS 0.8 (wgp.py's own default) -- TUNED FOR AUDIO, *NOT* VIDEO. #
REM #  MiniMax H3 VIDEO WILL OOM AT THIS DEFAULT. Override it per launch:     #
REM #                                                                         #
REM #    H3 Pruned 20B (fl2va/ref2va_pruned) .. set "VRAM_SAFETY=0.5"         #
REM #    H3 Full 33B   (fl2va/ref2va) ........ set "VRAM_SAFETY=0.35"         #
REM #                                          then 0.3, then PROFILE=5       #
REM #    (PROFILE=5 re-flattens audio to the legacy LM decoder -- see above)  #
REM #    MiniMax Music 3 / audio ............. 0.8 (the default) -- see below #
REM #                                                                         #
REM #  e.g.  set "VRAM_SAFETY=0.5" && mcp-server.bat                          #
REM ###########################################################################
REM
REM Why 0.8 for MiniMax Music 3: its bottleneck is an autoregressive Qwen3 stage
REM that runs one decode step per audio frame (25 fps -> 3000 steps for a 2 min
REM song). That encoder is 10.2 GB even at int8, so at VRAM_SAFETY=0.5 (~8 GB of
REM preloaded weights on this 16 GB card) it cannot stay resident and streams
REM from system RAM on EVERY step. 0.8 (~12.8 GB) is enough to hold it. The H3
REM reason for lowering the knob -- an OOM while VAE-encoding a reference image
REM -- does not apply to music: there is no image VAE encode on that path.
REM
REM Measured before this change (VRAM_SAFETY=0.5): ~32.6 s of GPU per 1 s of
REM song, linear -- 30 s took ~17.5 min, 120 s took 1h08m. Re-measure after.
REM ---------------------------------------------------------------------------
if "%VRAM_SAFETY%"==""   set VRAM_SAFETY=0.8
if "%PERC_RESERVED%"=="" set PERC_RESERVED=0.45
REM No default: an unset PROFILE means "let WanGP choose per output type".
set "PROFILE_ARG="
if not "%PROFILE%"=="" set "PROFILE_ARG=--profile %PROFILE%"

call venv\Scripts\activate.bat

REM Make sure the outputs folder exists before serving it
if not exist outputs mkdir outputs

REM Run the MCP server (which also serves /files/*) in the foreground
python wgp.py --mcp --mcp-transport streamable-http --mcp-host %MCP_HOST% --mcp-port %MCP_PORT% --mcp-console-output --mcp-allow-read-file-system --mcp-api-version 1 %PROFILE_ARG% --vram-safety-coefficient %VRAM_SAFETY% --perc-reserved-mem-max %PERC_RESERVED%
