# Build argument for base image selection
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# ==============================================================================
# Stage 1: Base Environment & Dependency Setup
# ==============================================================================
FROM ${BASE_IMAGE} AS base

ARG COMFYUI_VERSION=0.34.0
ARG CUDA_VERSION_FOR_COMFY=12.8
ARG ENABLE_PYTORCH_UPGRADE=false
ARG PYTORCH_INDEX_URL

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV PYTHONUNBUFFERED=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python 3.12, C++ compilation tools (for insightface wheel compilation), and media libs
RUN apt-get update && apt-get install -y \
    build-essential \
    python3.12 \
    python3.12-dev \
    git \
    wget \
    unzip \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    openssh-server \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install uv package manager and create virtual environment
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

RUN uv pip install comfy-cli==1.13.0 pip setuptools wheel cython "numpy<2.0.0"

# Install ComfyUI Core
RUN if [ -n "${CUDA_VERSION_FOR_COMFY}" ]; then \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --cuda-version "${CUDA_VERSION_FOR_COMFY}" --nvidia; \
    else \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia; \
    fi

# Clone the PuLID Flux Custom Node Extension
RUN git clone https://github.com/lldacing/ComfyUI_PuLID_Flux_ll.git /comfyui/custom_nodes/ComfyUI_PuLID_Flux_ll

# Upgrade PyTorch if requested
RUN if [ "$ENABLE_PYTORCH_UPGRADE" = "true" ]; then \
      uv pip install --force-reinstall torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL}; \
    fi

# Install PyTorch cu128 and node requirements (including insightface, facenet-pytorch, and facexlib)
RUN uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
      --index-url https://download.pytorch.org/whl/cu128 \
    && uv pip install -r /comfyui/requirements.txt \
    && for r in /comfyui/custom_nodes/*/requirements.txt; do \
         [ -f "$r" ] && uv pip install -r "$r" || true; \
       done \
    && uv pip install "numpy<2.0.0" onnxruntime-gpu insightface open-clip-torch facexlib \
    && uv pip install facenet-pytorch --no-deps \
    && uv pip install "transformers>=4.50.3,<5" "huggingface-hub<1.0" \
    && uv pip install runpod requests websocket-client

WORKDIR /comfyui
ADD src/extra_model_paths.yaml ./
WORKDIR /

ADD src/start.sh src/network_volume.py handler.py test_input.json ./
RUN chmod +x /start.sh

COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /usr/local/bin/comfy-node-install

ENV PIP_NO_INPUT=1

COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode

# ==============================================================================
# Stage 2: Pre-fetch Models and Weights (Avoid Cold Start Latency)
# ==============================================================================
FROM base AS downloader

ARG HUGGINGFACE_ACCESS_TOKEN
ARG MODEL_TYPE=fluxed-up

WORKDIR /comfyui

# Create model directory structure
RUN mkdir -p models/checkpoints \
             models/vae \
             models/unet \
             models/clip \
             models/text_encoders \
             models/diffusion_models \
             models/model_patches \
             models/pulid \
             models/insightface/models/antelopev2 \
             models/facexlib \
             /root/.cache/facexlib/weights/

# 1. Download PuLID Flux Adapter Weights
RUN wget -q -O models/pulid/PuLID-FLUX-v0.9.1.safetensors \
    https://huggingface.co/guozinan/PuLID/resolve/main/PuLID-FLUX-v0.9.1.safetensors

# 2. Download EVA-CLIP Model Weight for PuLID
RUN wget -q -O models/clip/EVA02_CLIP_L_336_psz14_s6B.pt \
    https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA02_CLIP_L_336_psz14_s6B.pt

# 3. Download and Extract InsightFace antelopev2 Models
RUN wget -q -O /tmp/antelopev2.zip https://huggingface.co/MONA-LISA/antelopev2/resolve/main/antelopev2.zip \
    && unzip /tmp/antelopev2.zip -d models/insightface/models/antelopev2 \
    && rm /tmp/antelopev2.zip

# 4. Download facexlib detection weights (prevents runtime download timeouts)
RUN wget -q -O /root/.cache/facexlib/weights/detection_Resnet50_Final.pth \
    https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth \
    && cp /root/.cache/facexlib/weights/detection_Resnet50_Final.pth models/facexlib/

# Conditional model downloads based on MODEL_TYPE
RUN if [ "$MODEL_TYPE" = "sdxl" ]; then \
      wget -q -O models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors && \
      wget -q -O models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors && \
      wget -q -O models/vae/sdxl-vae-fp16-fix.safetensors https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "sd3" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/checkpoints/sd3_medium_incl_clips_t5xxlfp8.safetensors https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "flux1-schnell" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/flux1-schnell.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors && \
      wget -q -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -q -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "flux1-dev" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/flux1-dev.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors && \
      wget -q -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -q -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "flux1-dev-fp8" ]; then \
      wget -q -O models/checkpoints/flux1-dev-fp8.safetensors https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "z-image-turbo" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/text_encoders/qwen_3_4b.safetensors https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors && \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/diffusion_models/z_image_turbo_bf16.safetensors https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors && \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors && \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/model_patches/Z-Image-Turbo-Fun-Controlnet-Union.safetensors https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union/resolve/main/Z-Image-Turbo-Fun-Controlnet-Union.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "fluxed-up" ]; then \
      wget -O models/checkpoints/fluxedUp.safetensors "https://huggingface.co/HurdyThirty/FluxedUp/resolve/main/fluxedUpFluxNSFW_40DevFp8.safetensors" && \
      wget -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors; \
    fi

# ==============================================================================
# Stage 3: Final Image Assembly and Import Verification
# ==============================================================================
FROM base AS final

# Transfer full model cache from Stage 2
COPY --from=downloader /comfyui/models /comfyui/models
COPY --from=downloader /root/.cache /root/.cache

# Symlink InsightFace home folder so runtime lookup passes
RUN mkdir -p /root/.insightface \
    && ln -s /comfyui/models/insightface/models /root/.insightface/models

# Build-time import check: Runs AFTER models & custom nodes are mounted
RUN cd /comfyui && timeout 300 python main.py --quick-test-for-ci --cpu

CMD ["/start.sh"]
