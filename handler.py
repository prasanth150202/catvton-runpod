"""RunPod Serverless handler for CatVTON virtual try-on.

Input (job["input"]):
    person_image   (str, required)  URL or base64 (optionally a data: URI)
    cloth_image    (str, required)  URL or base64
    cloth_type     (str)  "upper" | "lower" | "overall"   default "upper"
    mask           (str)  optional URL/base64 mask; white = area to repaint.
                          If omitted, the mask is generated automatically.
    num_inference_steps (int)  default 50
    guidance_scale (float)     default 2.5
    seed           (int)       default 42, -1 for random
    width, height  (int)       default 768 x 1024
    return_mask    (bool)      default False, also return the mask used

Output:
    {"image": "<base64 PNG>", "mask": "<base64 PNG>"?, "seed": int}
"""
import base64
import io
import os

import numpy as np
import requests
import runpod
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

from model.cloth_masker import AutoMasker
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

BASE_MODEL = os.environ.get("BASE_MODEL", "booksforcharlie/stable-diffusion-inpainting")
RESUME_PATH = os.environ.get("RESUME_PATH", "zhengchong/CatVTON")
MIXED_PRECISION = os.environ.get("MIXED_PRECISION", "bf16")
SKIP_SAFETY_CHECK = os.environ.get("SKIP_SAFETY_CHECK", "false").lower() == "true"
CLOTH_TYPES = ("upper", "lower", "overall")

# Loaded once per worker (outside the handler) so warm requests skip model loading.
repo_path = snapshot_download(repo_id=RESUME_PATH)
pipeline = CatVTONPipeline(
    base_ckpt=BASE_MODEL,
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype(MIXED_PRECISION),
    use_tf32=True,
    skip_safety_check=SKIP_SAFETY_CHECK,
    device="cuda",
)
mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device="cuda",
)


def load_image(value, mode="RGB"):
    if value.startswith(("http://", "https://")):
        resp = requests.get(value, timeout=30)
        resp.raise_for_status()
        data = resp.content
    else:
        if value.startswith("data:"):
            value = value.split(",", 1)[1]
        data = base64.b64decode(value)
    return Image.open(io.BytesIO(data)).convert(mode)


def to_base64(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def handler(job):
    params = job["input"]
    if not params.get("person_image") or not params.get("cloth_image"):
        return {"error": "person_image and cloth_image are required"}

    cloth_type = params.get("cloth_type", "upper")
    if cloth_type not in CLOTH_TYPES:
        return {"error": f"cloth_type must be one of {CLOTH_TYPES}"}

    width = int(params.get("width", 768))
    height = int(params.get("height", 1024))
    seed = int(params.get("seed", 42))
    if seed == -1:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    generator = torch.Generator(device="cuda").manual_seed(seed)

    try:
        person_image = resize_and_crop(load_image(params["person_image"]), (width, height))
        cloth_image = resize_and_padding(load_image(params["cloth_image"]), (width, height))

        if params.get("mask"):
            mask = np.array(load_image(params["mask"], mode="L"))
            mask[mask > 0] = 255
            mask = resize_and_crop(Image.fromarray(mask), (width, height))
        else:
            mask = automasker(person_image, cloth_type)["mask"]
        mask = mask_processor.blur(mask, blur_factor=9)

        result_image = pipeline(
            image=person_image,
            condition_image=cloth_image,
            mask=mask,
            num_inference_steps=int(params.get("num_inference_steps", 50)),
            guidance_scale=float(params.get("guidance_scale", 2.5)),
            height=height,
            width=width,
            generator=generator,
        )[0]
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    output = {"image": to_base64(result_image), "seed": seed}
    if params.get("return_mask"):
        output["mask"] = to_base64(mask)
    return output


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
