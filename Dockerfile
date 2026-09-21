# CatVTON on RunPod Serverless. Build for linux/amd64:
#   docker build --platform linux/amd64 -t <dockerhub-user>/catvton-runpod:latest .
FROM python:3.9-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 3.9 matches the prebuilt detectron2/_C.cpython-39 extension shipped in the repo.
# Upgrade pip first: 23.0.1 in the base image rejects non-normalized names on the PyTorch index.
# PyPI stays the main index; the cu121 index only supplies torch/torchvision (+cu121 wins over PyPI's build).
RUN pip install --upgrade pip \
    && pip install torch==2.4.0 torchvision==0.19.0 --extra-index-url https://download.pytorch.org/whl/cu121

# Pin diffusers instead of tracking git main (main no longer supports Python 3.9); peft is unused.
COPY requirements.txt .
RUN sed -e 's#^git+https://github.com/huggingface/diffusers.git#diffusers==0.31.0#' \
        -e '/^torch/d' -e '/^peft/d' -e '/^gradio/d' -e '/^setuptools/d' requirements.txt > req-runpod.txt \
    && pip install -r req-runpod.txt runpod requests

COPY builder/download_weights.py builder/download_weights.py
RUN python builder/download_weights.py

COPY . .

# Everything is cached in the image; never hit the Hub at runtime.
ENV HF_HUB_OFFLINE=1

CMD ["python", "-u", "handler.py"]
