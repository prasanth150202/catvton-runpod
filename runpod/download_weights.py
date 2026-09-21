"""Bake all model weights into the image's HF cache at build time, so cold starts don't download anything."""
import os

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from huggingface_hub import snapshot_download
from transformers import CLIPImageProcessor

BASE_MODEL = os.environ.get("BASE_MODEL", "booksforcharlie/stable-diffusion-inpainting")
RESUME_PATH = os.environ.get("RESUME_PATH", "zhengchong/CatVTON")

# Load the same way model/pipeline.py does, so exactly the files it needs end up in the cache.
DDIMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
CLIPImageProcessor.from_pretrained(BASE_MODEL, subfolder="feature_extractor")
StableDiffusionSafetyChecker.from_pretrained(BASE_MODEL, subfolder="safety_checker")
UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet")
AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
snapshot_download(repo_id=RESUME_PATH)
print("All weights downloaded.")
