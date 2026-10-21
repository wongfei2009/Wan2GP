# Fork-only (wongfei2009/Wan2GP). MangaLineExtraction (https://github.com/ljsabc/MangaLineExtraction_PyTorch,
# MIT, see LICENSE) as comfyui_controlnet_aux's "Manga Lineart" node runs it; model_torch.py is verbatim.
import cv2
import numpy as np
import torch

from .model_torch import res_skip


class MangaLineDetector:
    def __init__(self, weights_path, device="cpu"):
        self.model = res_skip()
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict({key.replace("module.", ""): value for key, value in state.items()})
        self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, image, detect_resolution):
        """image: HxWx3 uint8 RGB. Returns white lines on black, uint8, at the resized (unpadded) size.

        detect_resolution sizes the SHORT side, rounded up to a multiple of 256, as the node does;
        the MangaScreen workflow feeds it the image WIDTH, so a 1728x1152 page runs at 2688x1792."""
        resolution = 256 * int(np.ceil(float(detect_resolution) / 256.0))
        h, w = image.shape[:2]
        k = resolution / min(h, w)
        ht, wt = int(np.round(h * k)), int(np.round(w * k))
        resized = cv2.resize(image, (wt, ht), interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
        padded = np.pad(resized, [[0, -ht % 64], [0, -wt % 64], [0, 0]], mode="edge")
        gray = cv2.cvtColor(padded, cv2.COLOR_RGB2GRAY)
        line = self.model(torch.from_numpy(gray).float()[None, None].to(self.device))[0, 0].float().cpu().numpy()
        return 255 - np.clip(line, 0, 255).astype(np.uint8)[:ht, :wt]

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self
