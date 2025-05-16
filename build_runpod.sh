#!/usr/bin/env bash
set -euo pipefail

#── 1) Must run as root ───────────────────────────────────────
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (sudo)." >&2
  exit 1
fi

#── 2) Install Docker CE if missing ──────────────────────────
if ! command -v docker &>/dev/null; then
  echo "Installing Docker CE..."
  apt-get update -y
  apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

  mkdir -p /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" \
    > /etc/apt/sources.list.d/docker.list

  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
  if command -v systemctl &>/dev/null; then
    systemctl enable docker && systemctl start docker
  else
    echo "ⓘ systemctl not available, skipping docker service start."
  fi
else
  echo "Docker already installed."
fi

#── 3) Install NVIDIA Container Toolkit ────────────────────────
if ! dpkg -l | grep -q nvidia-container-toolkit; then
  echo "Installing NVIDIA Container Toolkit..."
  distribution="$(. /etc/os-release; echo $ID$VERSION_ID)"
  curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
  curl -fsSL "https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list" \
    | tee /etc/apt/sources.list.d/nvidia-docker.list

  apt-get update -y
  apt-get install -y nvidia-container-toolkit
  if command -v systemctl &>/dev/null; then
    systemctl restart docker
  else
    echo "ⓘ systemctl not available, docker will pick up nvidia-container-toolkit on next run."
  fi

#── 4) Clone (or update) runpod-workers/worker-vllm ────────────
WORKDIR="/workspace/worker-vllm-lora"
if [ ! -d "$WORKDIR" ]; then
  git clone https://github.com/AJAlkaline/worker-vllm-lora.git "$WORKDIR"
else
  echo "Updating existing repo in $WORKDIR"
  git -C "$WORKDIR" pull
fi

cd "$WORKDIR"

#── 5) Prepare Hugging Face secret ────────────────────────────
HF_TOKEN_FILE="${HOME}/.hf_token"
if [ ! -f "$HF_TOKEN_FILE" ]; then
  echo "Error: Please create $HF_TOKEN_FILE containing your HF API token." >&2
  exit 1
fi

#── 6) Enable BuildKit for secrets ────────────────────────────
export DOCKER_BUILDKIT=1

#── 7) Define your model & adapter ────────────────────────────
MODEL="unsloth/Mistral-Small-Instruct-2409-bnb-4bit"
LORA="alleavitch/hobielorafull"
# inside the image we’ll merge into /models
BASE_PATH="/models"
IMAGE="ajalkaline/worker-vllm-lora:latest"

#── 8) Build the Docker image ─────────────────────────────────
echo "Building $IMAGE with base=$MODEL + LoRA=$LORA …"
docker build \
  --secret id=HF_TOKEN,src="$HF_TOKEN_FILE" \
  --build-arg MODEL_NAME="$MODEL" \
  --build-arg LORA_NAME="$LORA" \
  --build-arg BASE_PATH="$BASE_PATH" \
  -t "$IMAGE" \
  .

#── 9) (Optional) push to registry ────────────────────────────
if [ "${DOCKER_PUSH:-1}" -eq 1 ]; then
  echo "Pushing $IMAGE to registry…"
  docker push "$IMAGE"
else
  echo "Skipping docker push (DOCKER_PUSH=0)."
fi

echo "✅ Done! Your RunPod‑ready image is: $IMAGE"
