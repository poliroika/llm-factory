cat > Dockerfile << 'EOF'
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl git ca-certificates build-essential \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv \
 && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
 && python3.12 -m pip install -U pip setuptools wheel

RUN python3.12 -m pip install \
    "torch>=2.2.0" \
    "transformers>=4.44.0" \
    "datasets>=2.19.0" \
    "peft>=0.12.0" \
    "trl>=0.9.6" \
    "accelerate>=0.33.0" \
    "tqdm>=4.66.0" \
    "rich>=13.0.0" \
    "numpy>=1.24.0"

COPY . /workspace/
EOF
