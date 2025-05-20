FROM nvidia/cuda:12.1.0-base-ubuntu22.04
# System dependencies
RUN apt-get update -y && \
    apt-get install -y python3-pip libgl1
# CUDA compatibility
RUN ldconfig /usr/local/cuda-12.1/compat/
# Python environment
RUN mkdir -p /builder
COPY requirements.txt /builder/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /builder/requirements.txt
# Quantization dependencies
RUN python3 -m pip install \
    vllm==0.8.5 \
    flashinfer \
    bitsandbytes>=0.45.3 -i https://flashinfer.ai/whl/cu121/torch2.3 \
    huggingface_hub
# Model configuration
ARG BASE_PATH="/model"
ARG MODEL_ID="scb10x/typhoon2.1-gemma3-4b"
ARG HF_TOKEN=hf_YOUR_TOKEN_HERE
ENV MODEL_NAME=$MODEL_ID \
    VLLM_MODEL_PATH=$BASE_PATH \
    VLLM_DTYPE="bfloat16" \
    QUANTIZATION="bitsandbytes" \
    HF_HOME="$BASE_PATH/huggingface-cache" \
    VLLM_GPU_MEMORY_UTILIZATION=0.95 \
    VLLM_MAX_MODEL_LEN=4096
# Create model and src directories
RUN mkdir -p $BASE_PATH
RUN mkdir -p /src
# Download model with efficient caching
RUN --mount=type=cache,target=/root/.cache/huggingface \
    HF_HOME=/root/.cache/huggingface \
    python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='$MODEL_ID', \
    local_dir='$BASE_PATH', \
    token='$HF_TOKEN', \
    force_download=False, \
    resume_download=True)"
# Copy source code if exists
COPY src/* /src/ 2>/dev/null || :
# Simplified command without parameters
CMD ["python3", "/src/handler.py"]
