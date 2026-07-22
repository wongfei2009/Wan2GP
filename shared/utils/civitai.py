"""Civitai URL resolution via the public (auth-free) metadata API.

Actual file downloads usually require an account API key; that is handled at
download time (shared/utils/download.py appends ?token=... from
CIVITAI_API_TOKEN for civitai.com URLs). Metadata lookups never need it.

Standalone module: no WanGP runtime imports, so it can be unit-tested anywhere.
"""

import json
import re
import urllib.parse
import urllib.request

USER_AGENT = "Mozilla/5.0 (compatible; WanGP)"
_API_BASE = "https://civitai.com/api/v1"
_HOSTS = ("civitai.com", "www.civitai.com")
# Windows-illegal filename characters plus control chars.
_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def is_civitai_url(url):
    try:
        host = urllib.parse.urlsplit(url).hostname or ""
    except ValueError:
        return False
    return host.lower() in _HOSTS


def sanitize_filename(name, default_ext=".safetensors"):
    name = urllib.parse.unquote(str(name or ""))
    name = _ILLEGAL_CHARS.sub("", name).strip().strip(". ")
    if not name:
        raise ValueError("empty filename after sanitizing")
    if not name.lower().endswith((".safetensors", ".sft")):
        name += default_ext
    return name


def _get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def resolve_civitai_url(url):
    """Resolve a pasted Civitai URL to download info.

    Accepted shapes:
      https://civitai.com/api/download/models/<versionId>[?...]
      https://civitai.com/models/<modelId>[/<slug>][?modelVersionId=<versionId>]

    Returns None when the URL is not a Civitai URL (caller falls back to a
    direct download). Raises ValueError with a user-readable message when it
    is Civitai but cannot be resolved.

    Result: {download_url, filename, size_kb, trained_words, version_id,
             version_name, model_name, model_page}
    """
    if not is_civitai_url(url):
        return None
    split = urllib.parse.urlsplit(url)
    path_parts = [part for part in split.path.split("/") if part]
    query = urllib.parse.parse_qs(split.query)

    version_id = model_id = model_name = None
    if len(path_parts) >= 4 and path_parts[:3] == ["api", "download", "models"]:
        version_id = path_parts[3]
    elif len(path_parts) >= 2 and path_parts[0] == "models":
        model_id = path_parts[1]
        version_id = (query.get("modelVersionId") or [None])[0]
    else:
        raise ValueError(f"unrecognized Civitai URL shape: {url}")

    if version_id is None:
        model = _get_json(f"{_API_BASE}/models/{model_id}")
        versions = model.get("modelVersions") or []
        if not versions:
            raise ValueError(f"Civitai model {model_id} has no versions")
        version = versions[0]
        version_id = version.get("id")
        model_name = model.get("name")
    else:
        version = _get_json(f"{_API_BASE}/model-versions/{version_id}")
        model_name = (version.get("model") or {}).get("name")

    files = version.get("files") or []
    safetensors = [f for f in files if str(f.get("name") or "").lower().endswith((".safetensors", ".sft"))]
    model_files = [f for f in safetensors if f.get("type") == "Model"]
    candidates = [f for f in model_files if f.get("primary")] or model_files or safetensors
    if not candidates:
        raise ValueError(f"Civitai version {version_id} has no safetensors file")
    file = candidates[0]

    page_model_id = version.get("modelId") or model_id
    return {
        "download_url": file.get("downloadUrl") or f"https://civitai.com/api/download/models/{version_id}",
        "filename": sanitize_filename(file.get("name")),
        "size_kb": file.get("sizeKB"),
        "trained_words": version.get("trainedWords") or [],
        "version_id": version_id,
        "version_name": version.get("name"),
        "model_name": model_name,
        "model_page": f"https://civitai.com/models/{page_model_id}" if page_model_id else None,
    }
