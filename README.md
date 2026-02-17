# ComfyUI Face Parsing

A custom node pack for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that provides **face semantic segmentation** using the [jonathandinu/face-parsing](https://huggingface.co/jonathandinu/face-parsing) model (SegFormer-based). Parse faces into 19 distinct regions and generate precise masks for skin retouching, inpainting, makeup transfer, and more.

---

## ✨ Features

- **Automatic model downloading** — All required models are fetched from HuggingFace on first run
- **19 individually toggleable face regions** — Skin, nose, eyes, eyebrows, lips, ears, hair, hat, neck, cloth, and more
- **GPU acceleration** — Run inference on CUDA or CPU
- **Batch processing** — Process multiple images in a single pass
- **Colorized preview** — Visual segmentation map output for debugging

---

## 📦 Installation

### Option 1: Manual Install

1. Clone or copy this folder into your ComfyUI `custom_nodes/` directory:
   ```bash
   cp -r comfyui_face_parsing /path/to/ComfyUI/custom_nodes/
   ```
2. Restart ComfyUI
3. Models will auto-download on first load (~160MB total)

### Option 2: Symlink (Development)

```bash
ln -s /path/to/comfyui_face_parsing /path/to/ComfyUI/custom_nodes/comfyui_face_parsing
```

---

## 📋 Dependencies

Specified in `requirements.txt` and auto-installed on first load:

| Package | Purpose |
|---------|---------|
| `transformers` | SegFormer model and image processor |
| `ultralytics` | YOLOv8 face detection model |

Additional dependencies already included with ComfyUI: `torch`, `torchvision`, `numpy`, `matplotlib`, `Pillow`.

---

## 🧩 Nodes

All nodes appear under the **`face_parsing`** category in the ComfyUI add-node menu.

### FaceParsingModelLoaderNew

Loads the SegFormer semantic segmentation model from the local model directory.

| | Type | Description |
|---|------|-------------|
| **Input** | `device` (dropdown) | `cpu` or `cuda` — where to load the model |
| **Output** | `FACE_PARSING_MODEL` | The loaded segmentation model |

---

### FaceParsingProcessorLoader

Loads the SegformerImageProcessor used to preprocess images before inference.

| | Type | Description |
|---|------|-------------|
| **Input** | *(none)* | No inputs required |
| **Output** | `FACE_PARSING_PROCESSOR` | The loaded image processor |

---

### FaceParse

Runs face parsing inference on an input image. Produces both a colorized segmentation visualization and raw parsing results for downstream mask generation.

| | Type | Description |
|---|------|-------------|
| **Input** | `model` (`FACE_PARSING_MODEL`) | From FaceParsingModelLoader |
| **Input** | `processor` (`FACE_PARSING_PROCESSOR`) | From FaceParsingProcessorLoader |
| **Input** | `image` (`IMAGE`) | The image to parse |
| **Output** | `IMAGE` | Colorized segmentation map (viridis colormap) |
| **Output** | `FACE_PARSING_RESULT` | Raw segmentation tensor for mask generation |

---

### FaceParsingResultsParser

Converts raw face parsing results into a binary mask. Each of the 19 face regions can be individually toggled on/off to create a combined mask.

| | Type | Description |
|---|------|-------------|
| **Input** | `result` (`FACE_PARSING_RESULT`) | From FaceParse node |
| **Input** | 19× `BOOLEAN` toggles | Enable/disable each face region |
| **Output** | `MASK` | Combined binary mask of selected regions |

#### Face Region Toggles

| Index | Region | Default | Index | Region | Default |
|-------|--------|---------|-------|--------|---------|
| 0 | `background` | ❌ Off | 10 | `mouth` | ✅ On |
| 1 | `skin` | ✅ On | 11 | `u_lip` (upper lip) | ✅ On |
| 2 | `nose` | ✅ On | 12 | `l_lip` (lower lip) | ✅ On |
| 3 | `eye_g` (eyeglasses) | ✅ On | 13 | `hair` | ✅ On |
| 4 | `r_eye` (right eye) | ✅ On | 14 | `hat` | ✅ On |
| 5 | `l_eye` (left eye) | ✅ On | 15 | `ear_r` (earring) | ✅ On |
| 6 | `r_brow` (right eyebrow) | ✅ On | 16 | `neck_l` (necklace) | ✅ On |
| 7 | `l_brow` (left eyebrow) | ✅ On | 17 | `neck` | ✅ On |
| 8 | `r_ear` (right ear) | ✅ On | 18 | `cloth` (clothing) | ✅ On |
| 9 | `l_ear` (left ear) | ✅ On | | | |

---

## 🔗 Typical Workflow

```
FaceParsingModelLoader ──(FACE_PARSING_MODEL)──┐
                                                ▼
FaceParsingProcessorLoader ──(FACE_PARSING_PROCESSOR)──► FaceParse ──(FACE_PARSING_RESULT)──► FaceParsingResultsParser ──(MASK)──►
                                                            ▲
                                           Input IMAGE ─────┘
```

### Example Use Cases

- **Skin retouching** — Enable only `skin`, disable everything else → use mask for inpainting/denoising
- **Hair color change** — Enable only `hair` → use mask to isolate and recolor hair
- **Lip makeup** — Enable `u_lip` + `l_lip` → apply color grading or style transfer to lips
- **Background removal** — Enable `background` → invert mask to keep only the face
- **Face-only inpainting** — Disable `background`, `hair`, `cloth`, `hat` → mask covers only facial features

---

## 📁 Downloaded Models

On first load, the following models are automatically downloaded:

| Model | Source | Location |
|-------|--------|----------|
| `model.safetensors` | [jonathandinu/face-parsing](https://huggingface.co/jonathandinu/face-parsing) | `models/face_parsing/` |
| `config.json` | [jonathandinu/face-parsing](https://huggingface.co/jonathandinu/face-parsing) | `models/face_parsing/` |
| `preprocessor_config.json` | [jonathandinu/face-parsing](https://huggingface.co/jonathandinu/face-parsing) | `models/face_parsing/` |
| `face_yolov8m.pt` | [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer) | `models/ultralytics/bbox/` |

---

## 📂 File Structure

```
comfyui_face_parsing/
├── __init__.py              # Model downloads, dependency checks, node export
├── face_parsing_nodes.py    # All 4 node class implementations
├── requirements.txt         # Python dependencies (transformers, ultralytics)
└── README.md                # This file
```

---

## 🙏 Credits

- **Model**: [jonathandinu/face-parsing](https://huggingface.co/jonathandinu/face-parsing) — SegFormer fine-tuned on CelebAMask-HQ
- **Face Detection**: [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer) — YOLOv8 face detection
- **Original Inspiration**: [Ryuukeisyou/comfyui_face_parsing](https://github.com/Ryuukeisyou/comfyui_face_parsing)

---

## 📄 License

MIT License
