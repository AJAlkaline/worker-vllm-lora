#!/usr/bin/env bash
set -euo pipefail

#———————————————————————————————————————————————————————————————————————————————
# 1) Check required env vars
#———————————————————————————————————————————————————————————————————————————————
: "${DOCKER_USER:?Need to set DOCKER_USER (your Docker Hub username)}"
: "${DOCKER_TOKEN:?Need to set DOCKER_TOKEN (your Docker Hub access token)}"
: "${MODEL_NAME:?Need to set MODEL_NAME (e.g. unsloth/Mistral-Small-Instruct-2409-bnb-4bit)}"
: "${LORA_NAME:?Need to set LORA_NAME (your LoRA repo or path)}"

#———————————————————————————————————————————————————————————————————————————————
# 2) Install system deps
#———————————————————————————————————————————————————————————————————————————————
apt-get update -y
apt-get install -y sudo git curl wget unzip build-essential

#———————————————————————————————————————————————————————————————————————————————
# 3) Install Docker CE via get.docker.com
#———————————————————————————————————————————————————————————————————————————————
if ! command -v docker &>/dev/null; then
  curl -fsSL https://get.docker.com -o get-docker.sh
  sudo sh get-docker.sh
else
  echo "Docker already installed"
fi

#———————————————————————————————————————————————————————————————————————————————
# 4) Login to Docker Hub
#———————————————————————————————————————————————————————————————————————————————
echo "$DOCKER_TOKEN" | docker login -u "$DOCKER_USER" --password-stdin

#———————————————————————————————————————————————————————————————————————————————
# 5) Install Bazelisk (Bazel launcher)
#———————————————————————————————————————————————————————————————————————————————
if ! command -v bazel &>/dev/null; then
  wget -qO /usr/local/bin/bazel \
    https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64
  chmod +x /usr/local/bin/bazel
else
  echo "Bazel already installed"
fi

#———————————————————————————————————————————————————————————————————————————————
# 6) Clone or update your worker-vllm-lora repo
#———————————————————————————————————————————————————————————————————————————————
WORKDIR="/workspace/worker-vllm-lora"
if [ ! -d "$WORKDIR" ]; then
  git clone https://github.com/AJAlkaline/worker-vllm-lora.git "$WORKDIR"
else
  echo "Updating existing repo"
  git -C "$WORKDIR" pull
fi
cd "$WORKDIR"

#———————————————————————————————————————————————————————————————————————————————
# 7) Build & push with Bazel
#———————————————————————————————————————————————————————————————————————————————
echo "🛠️  Running Bazel to build & push the image…"
bazel run //:push_custom_image \
  --define="MODEL_NAME=$MODEL_NAME" \
  --define="LORA_NAME=$LORA_NAME"

echo "🎉  Done! Your image is at: docker.io/${DOCKER_USER}/worker-vllm-lora:latest"
