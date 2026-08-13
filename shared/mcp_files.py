"""HTTP file routes for WanGP's MCP server (fork-only).

Folds the outputs file server -- previously a separate `uploadserver` process
on its own port, started by mcp-server.bat -- into the MCP server's own
Starlette app, so ONE process serves both JSON-RPC and media:

    GET  /files/<relpath>   download a file from the outputs directory
    GET  /files/<dir>/      plain-text listing, one name per line ("dir/" for
                            subdirectories); "" lists the outputs root
    POST /files/upload      multipart upload (form field "files", repeatable)
                            into the outputs root; an existing file with the
                            same name is replaced, matching the semantics of
                            `uploadserver --allow-replace` so clients can keep
                            predicting the uploaded path

Trust model is unchanged from the old two-process setup: NO AUTH on any route,
exposure bounded only by the firewall. The path guard below keeps requests
inside the outputs directory (traversal / symlink escapes); it is not
authentication.

Known quirk, inherited deliberately: a file literally named "upload" at the
outputs root is shadowed by the upload route for POST, but still downloadable
via GET (the routes differ by method).
"""

from __future__ import annotations

from pathlib import Path


def register_file_routes(mcp, outputs_root: Path | str) -> None:
    """Register the /files/* routes on a FastMCP instance.

    Call only for HTTP transports; requires the official MCP SDK's
    FastMCP.custom_route (present in every SDK recent enough to speak the
    streamable-http transport this server runs).
    """

    from starlette.requests import Request
    from starlette.responses import FileResponse, PlainTextResponse, Response

    root = Path(outputs_root).resolve()

    def _resolve(relpath: str) -> Path | None:
        # Resolve inside the outputs root only; reject .. traversal and
        # symlinks that point outside. resolve() also normalizes the
        # Windows/posix separator mix that URL path params can carry.
        candidate = (root / relpath).resolve()
        if candidate == root or root in candidate.parents:
            return candidate
        return None

    @mcp.custom_route("/files/upload", methods=["POST"])
    async def upload_files(request: Request) -> Response:
        # Starlette's multipart parsing needs python-multipart, which the
        # WanGP venv already carries as a Gradio dependency.
        form = await request.form()
        uploads = [item for item in form.getlist("files") if getattr(item, "filename", None)]
        if not uploads:
            return PlainTextResponse("multipart field 'files' missing or empty\n", status_code=400)
        root.mkdir(parents=True, exist_ok=True)
        for item in uploads:
            # Basename only: uploads land flat in the outputs root, never in
            # subdirectories, no matter what path the client sent.
            name = Path(str(item.filename).replace("\\", "/")).name
            if not name or name in (".", ".."):
                return PlainTextResponse(f"bad upload filename {item.filename!r}\n", status_code=400)
            dest = root / name
            with open(dest, "wb") as fh:
                while chunk := await item.read(1 << 20):
                    fh.write(chunk)
        # uploadserver replies 204 No Content on success; clients check for it.
        return Response(status_code=204)

    @mcp.custom_route("/files/{path:path}", methods=["GET"])
    async def download_file(request: Request) -> Response:
        target = _resolve(request.path_params.get("path", ""))
        if target is None:
            return PlainTextResponse("path escapes the outputs directory\n", status_code=403)
        if target.is_dir():
            try:
                names = sorted(
                    entry.name + ("/" if entry.is_dir() else "")
                    for entry in target.iterdir()
                )
            except OSError as exc:
                return PlainTextResponse(f"cannot list directory: {exc}\n", status_code=500)
            return PlainTextResponse("".join(name + "\n" for name in names))
        if not target.is_file():
            return PlainTextResponse("not found\n", status_code=404)
        return FileResponse(target)
