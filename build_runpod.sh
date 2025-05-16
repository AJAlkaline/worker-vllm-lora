#!/usr/bin/env bash
set -euo pipefail

#———————————————————————————————————————————————————————————————————————————————
# 1) Check required env vars
#———————————————————————————————————————————————————————————————————————————————
: "${DOCKER_USER:?Need to set DOCKER_USER}"
: "${DOCKER_TOKEN:?Need to set DOCKER_TOKEN}"

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
  wget -qO /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64
  chmod +x /usr/local/bin/bazel
else
  echo "Bazel already installed"
fi

#———————————————————————————————————————————————————————————————————————————————
# 6) Clone your worker-vllm-lora repo
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
# 7) Create Bazel WORKSPACE with rules_docker
#———————————————————————————————————————————————————————————————————————————————
cat > WORKSPACE <<'EOF'
workspace(name = "worker_vllm_lora")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# rules_docker
http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "fb5729a861d61896f8aeecc65c1dcaf149048393ca5bfb3da93735283cd8c7e3",
    strip_prefix = "rules_docker-0.28.0",
    urls = ["https://github.com/bazelbuild/rules_docker/archive/v0.28.0.tar.gz"],
)

load("@io_bazel_rules_docker//container:container.bzl", "container_repositories")
container_repositories()

load("@io_bazel_rules_docker//go:image.bzl", "go_image")
EOF

#———————————————————————————————————————————————————————————————————————————————
# 8) Create BUILD.bazel for our Dockerfile
#———————————————————————————————————————————————————————————————————————————————
cat > BUILD.bazel <<'EOF'
load("@io_bazel_rules_docker//container:container.bzl", "docker_build", "oci_push")

# Build the local image from your Dockerfile
docker_build(
    name = "custom_image",
    dockerfile = "Dockerfile",
    # copy entire workspace so download_model.py, src/, etc. are included
    directory = ".",
    tars = [],
)

# Push to Docker Hub
oci_push(
    name         = "push_custom_image",
    image        = ":custom_image",
    repository   = "index.docker.io/${DOCKER_USER}/worker-vllm-lora",
    remote_tags  = ["latest"],
)
EOF

#———————————————————————————————————————————————————————————————————————————————
# 9) Build & push with Bazel
#———————————————————————————————————————————————————————————————————————————————
echo "🛠️  Running Bazel to build & push the image…"
bazel run //:push_custom_image

echo "🎉  Done! Check https://hub.docker.com/r/${DOCKER_USER}/worker-vllm-lora:latest"
