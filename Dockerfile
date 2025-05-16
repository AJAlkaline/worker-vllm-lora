# 1) Base image with CUDA & Python
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# 2) Install OS packages + pip
RUN apt-get update -y && \
    apt-get install -y python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# 3) Install Python deps
COPY builder/requirements.txt /requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install -r /requirements.txt && \
    pip3 install \
      vllm==0.8.5 \
      bitsandbytes>=0.45.0 \
      peft \
      transformers \
      huggingface-hub

# 4) Build‑time args for your models & output path
ARG MODEL_NAME
ARG LORA_NAME
ARG BASE_PATH=/models

ENV BASE_PATH=${BASE_PATH} \
    HF_HOME=${BASE_PATH}/hf_cache \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

# 5) Copy in your vLLM worker code
COPY src /src

# 6) Copy the merge helper
COPY merge_lora.py /merge_lora.py

# 7) Download & merge the models at build time
RUN --mount=type=secret,id=HF_TOKEN \
    python3 /merge_lora.py \
      --base_model ${MODEL_NAME} \
      --lora_adapter ${LORA_NAME} \
      --output_dir ${BASE_PATH}

# 8) Write out local_model_args.json so vLLM will load /models offline
RUN python3 - << 'EOF'
import json, os
# This tells the worker to load the merged model from /models
with open("/local_model_args.json","w") as f:
    json.dump({"MODEL_NAME": "/models"}, f)
EOF

# 9) Final entrypoint
WORKDIR /
CMD ["python3", "/src/handler.py"]
