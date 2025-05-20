FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt-get update -y \
    && apt-get install -y python3-pip

RUN ldconfig /usr/local/cuda-12.1/compat/

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt
    
# Install vLLM (switching back to pip installs since issues that required building fork are fixed and space optimization is not as important since caching) and FlashInfer 
RUN python3 -m pip install vllm==0.8.5 && \
    python3 -m pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3
    

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME="scb10x/typhoon2.1-gemma3-4b"
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/model"
ARG QUANTIZATION="bitsandbytes"
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    MAX_MODEL_LEN = 4096 \
    LOAD_FORMAT = safetensors \
    DEVICE = cuda \
    QUANTIZATION=$QUANTIZATION \
    GPU_MEMORY_UTILIZATION = 0.95 \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 

ENV PYTHONPATH="/:/vllm-workspace"
# Download model using git-lfs
RUN mkdir -p $BASE_PATH && \
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL_ID', local_dir='$BASE_PATH', local_dir_use_symlinks=False, cache_dir='$HF_HOME')
    
# Application setup
COPY src /src

# Simplified command without parameters
CMD ["python3", "/src/handler.py"]
