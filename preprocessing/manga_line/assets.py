# Fork-only (wongfei2009/Wan2GP): weights for the manga line extractor used by wangp_manga_finish.
"""Download declaration shared by all consumers of these weights."""

WEIGHTS = "manga_line/erika.pth"


def query_download_def():
    # The same checkpoint comfyui_controlnet_aux's Manga Lineart node pulls.
    return {"repoId": "lllyasviel/Annotators", "sourceFolderList": [""], "fileList": [["erika.pth"]], "targetFolderList": ["manga_line"]}
