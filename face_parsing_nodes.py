import os
import numpy as np
import folder_paths
import torch
from torch import Tensor
import torch.nn as nn
import matplotlib
import matplotlib.colors
import torchvision.transforms as T

models_path = folder_paths.models_dir
face_parsing_path = os.path.join(models_path, "face_parsing")


class FaceParsingModelLoaderNew:
    """Loads the face parsing semantic segmentation model (jonathandinu/face-parsing)."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["cpu", "cuda"], {
                    "default": "cpu"
                })
            }
        }

    RETURN_TYPES = ("FACE_PARSING_MODEL",)
    FUNCTION = "main"
    CATEGORY = "face_parsing"

    def main(self, device: str):
        from transformers import AutoModelForSemanticSegmentation
        model = AutoModelForSemanticSegmentation.from_pretrained(face_parsing_path)
        if device == "cuda" and torch.cuda.is_available():
            model.cuda()
        return (model,)


class FaceParsingProcessorLoader:
    """Loads the SegformerImageProcessor for face parsing preprocessing."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {}
        }

    RETURN_TYPES = ("FACE_PARSING_PROCESSOR",)
    FUNCTION = "main"
    CATEGORY = "face_parsing"

    def main(self):
        from transformers import SegformerImageProcessor
        processor = SegformerImageProcessor.from_pretrained(face_parsing_path)
        return (processor,)


class FaceParse:
    """Runs face parsing inference on an image using the loaded model and processor.
    Outputs a colorized segmentation image and raw parsing results."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("FACE_PARSING_MODEL", {}),
                "processor": ("FACE_PARSING_PROCESSOR", {}),
                "image": ("IMAGE", {}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FACE_PARSING_RESULT",)
    FUNCTION = "main"
    CATEGORY = "face_parsing"

    def main(self, model, processor, image: Tensor):
        images = []
        results = []
        transform = T.ToPILImage()
        colormap = matplotlib.colormaps['viridis']
        device = model.device

        for item in image:
            size = item.shape[:2]
            inputs = processor(images=transform(item.permute(2, 0, 1)), return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=size,
                mode="bilinear",
                align_corners=False)

            pred_seg = upsampled_logits.argmax(dim=1)[0]
            pred_seg_np = pred_seg.cpu().detach().numpy().astype(np.uint8)
            results.append(torch.tensor(pred_seg_np))

            norm = matplotlib.colors.Normalize(0, 18)
            pred_seg_np_normed = norm(pred_seg_np)
            colored = colormap(pred_seg_np_normed)
            colored_sliced = colored[:, :, :3]
            images.append(torch.tensor(colored_sliced))

        images_out = torch.stack(images, dim=0)
        results_out = torch.stack(results, dim=0)
        return (images_out, results_out,)


class FaceParsingResultsParser:
    """Parses face parsing results into a selectable mask.
    Toggle individual face regions (skin, nose, eyes, lips, etc.) to create a combined mask."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "result": ("FACE_PARSING_RESULT", {}),
                "background": ("BOOLEAN", {"default": False}),
                "skin": ("BOOLEAN", {"default": True}),
                "nose": ("BOOLEAN", {"default": True}),
                "eye_g": ("BOOLEAN", {"default": True}),
                "r_eye": ("BOOLEAN", {"default": True}),
                "l_eye": ("BOOLEAN", {"default": True}),
                "r_brow": ("BOOLEAN", {"default": True}),
                "l_brow": ("BOOLEAN", {"default": True}),
                "r_ear": ("BOOLEAN", {"default": True}),
                "l_ear": ("BOOLEAN", {"default": True}),
                "mouth": ("BOOLEAN", {"default": True}),
                "u_lip": ("BOOLEAN", {"default": True}),
                "l_lip": ("BOOLEAN", {"default": True}),
                "hair": ("BOOLEAN", {"default": True}),
                "hat": ("BOOLEAN", {"default": True}),
                "ear_r": ("BOOLEAN", {"default": True}),
                "neck_l": ("BOOLEAN", {"default": True}),
                "neck": ("BOOLEAN", {"default": True}),
                "cloth": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "main"
    CATEGORY = "face_parsing"

    def main(
            self,
            result: Tensor,
            background: bool,
            skin: bool,
            nose: bool,
            eye_g: bool,
            r_eye: bool,
            l_eye: bool,
            r_brow: bool,
            l_brow: bool,
            r_ear: bool,
            l_ear: bool,
            mouth: bool,
            u_lip: bool,
            l_lip: bool,
            hair: bool,
            hat: bool,
            ear_r: bool,
            neck_l: bool,
            neck: bool,
            cloth: bool):

        # Mapping: region name -> segmentation class index
        region_map = [
            (background, 0),
            (skin, 1),
            (nose, 2),
            (eye_g, 3),
            (r_eye, 4),
            (l_eye, 5),
            (r_brow, 6),
            (l_brow, 7),
            (r_ear, 8),
            (l_ear, 9),
            (mouth, 10),
            (u_lip, 11),
            (l_lip, 12),
            (hair, 13),
            (hat, 14),
            (ear_r, 15),
            (neck_l, 16),
            (neck, 17),
            (cloth, 18),
        ]

        masks = []
        for item in result:
            mask = torch.zeros(item.shape, dtype=torch.uint8)
            for enabled, class_idx in region_map:
                if enabled:
                    mask = mask | torch.where(item == class_idx, 1, 0)
            masks.append(mask.float())

        mask_out = torch.stack(masks, dim=0)
        return (mask_out,)


# Node registration
NODE_CLASS_MAPPINGS = {
    'FaceParsingProcessorLoader(FaceParsing)': FaceParsingProcessorLoader,
    'FaceParsingModelLoaderNew(FaceParsing)': FaceParsingModelLoaderNew,
    'FaceParse(FaceParsing)': FaceParse,
    'FaceParsingResultsParser(FaceParsing)': FaceParsingResultsParser,
}
