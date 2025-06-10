from abc import abstractmethod

import cv2
import gdown
import numpy as np
import torch

from boxmot.appearance.reid.registry import ReIDModelRegistry
from boxmot.utils import logger as LOGGER
from boxmot.utils.checks import RequirementsChecker


class BaseModelBackend:
    def __init__(self, weights, device, half):
        self.weights = weights[0] if isinstance(weights, list) else weights
        self.device = device
        self.half = half
        self.model = None
        self.cuda = torch.cuda.is_available() and self.device.type != "cpu"

        self.download_model(self.weights)
        self.model_name = ReIDModelRegistry.get_model_name(self.weights)

        self.model = ReIDModelRegistry.build_model(
            self.model_name,
            num_classes=ReIDModelRegistry.get_nr_classes(self.weights),
            pretrained=not (self.weights and self.weights.is_file()),
            use_gpu=device,
        )
        self.checker = RequirementsChecker()
        self.load_model(self.weights)
        self.input_shape = (384, 128) if "lmbn" in self.model_name else (256, 128)

    def get_crops(self, xyxys, img):
        h, w = img.shape[:2]
        interpolation_method = cv2.INTER_LINEAR
        mean_array = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std_array = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        crops_list = []

        for box in xyxys:
            x1, y1, x2, y2 = box.round().astype("int")
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                print(f"⚠️ Skipping invalid crop: ({x1},{y1}) to ({x2},{y2})")
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                print(f"⚠️ Crop has zero size. Skipping.")
                continue

            crop = cv2.resize(
                crop,
                (self.input_shape[1], self.input_shape[0]),
                interpolation=interpolation_method,
            )
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            crop = torch.from_numpy(crop).to(
                self.device, dtype=torch.half if self.half else torch.float
            )
            crop = torch.permute(crop, (2, 0, 1))  # (C, H, W)
            crops_list.append(crop)

        if not crops_list:
            return torch.empty((0, 3, *self.input_shape), device=self.device)

        crops = torch.stack(crops_list)  # Shape: (N, 3, H, W)
        crops = crops / 255.0
        crops = (crops - mean_array) / std_array


        return crops


    @torch.no_grad()
    def get_features(self, xyxys, img):
        if xyxys.size != 0:
            crops = self.get_crops(xyxys, img)
            crops = self.inference_preprocess(crops)
            features = self.forward(crops)
            features = self.inference_postprocess(features)
        else:
            features = np.array([])
        features = features / np.linalg.norm(features, axis=-1, keepdims=True)
        return features

    def warmup(self, imgsz=[(256, 128, 3)]):
        # warmup model by running inference once
        if self.device.type != "cpu":
            im = np.random.randint(0, 255, *imgsz, dtype=np.uint8)
            crops = self.get_crops(
                xyxys=np.array([[0, 0, 64, 64], [0, 0, 128, 128]]), img=im
            )
            crops = self.inference_preprocess(crops)
            self.forward(crops)  # warmup

    def to_numpy(self, x):
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

    def inference_preprocess(self, x):
        if self.half:
            if isinstance(x, torch.Tensor):
                if x.dtype != torch.float16:
                    x = x.half()
            elif isinstance(x, np.ndarray):
                if x.dtype != np.float16:
                    x = x.astype(np.float16)

        if self.nhwc:
            if isinstance(x, torch.Tensor):
                x = x.permute(0, 2, 3, 1)  # Convert from NCHW to NHWC
            elif isinstance(x, np.ndarray):
                x = np.transpose(x, (0, 2, 3, 1))  # Convert from NCHW to NHWC
        return x

    def inference_postprocess(self, features):
        if isinstance(features, (list, tuple)):
            return (
                self.to_numpy(features[0]) if len(features) == 1 else [self.to_numpy(x) for x in features]
            )
        else:
            return self.to_numpy(features)

    @abstractmethod
    def forward(self, im_batch):
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def load_model(self, w):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def download_model(self, w):
        if w.suffix == ".pt":
            model_url = ReIDModelRegistry.get_model_url(w)
            if not w.exists() and model_url is not None:
                gdown.download(model_url, str(w), quiet=False)
            elif not w.exists():
                LOGGER.error(
                    f"No URL associated with the chosen StrongSORT weights ({w}). Choose between:"
                )
                ReIDModelRegistry.show_downloadable_models()
                exit()
