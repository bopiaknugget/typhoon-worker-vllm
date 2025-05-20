FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# System dependencies
RUN apt-get update -y && \
    apt-get install -y python3-pip libgl1 git git-lfs curl

# CUDA compatibility
RUN ldconfig /usr/local/cuda-12.1/compat/

# Python environment
COPY builder/requirements.txt /requirements.txt

RUN python3 -m pip install vllm==0.8.5 && \
    python3 -m pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3


# Quantization dependenci

# Model configuration
ARG BASE_PATH="/model"
ARG MODEL_ID="scb10x/typhoon2.1-gemma3-4b"
ENV MODEL_NAME=$MODEL_ID \
    VLLM_MODEL_PATH=$BASE_PATH \
    VLLM_DTYPE="bfloat16" \
    QUANTIZATION="bitsandbytes" \
    HF_HOME="$BASE_PATH/huggingface-cache" \
    VLLM_GPU_MEMORY_UTILIZATION=0.95 \
    VLLM_MAX_MODEL_LEN=4096 \
    HUGGING_FACE_HUB_TOKEN=hf_YOUR_TOKEN_HERE

# Download model using git-lfs
RUN mkdir -p $BASE_PATH && \
    git lfs install && \
    git lfs clone https://huggingface.co/$MODEL_ID $BASE_PATH

# Application setup
COPY src /src

# Simplified command without parameters
CMD ["python3", "/src/handler.py"]
