# consider adding the following:
    # apt-get update && apt-get install -y xvfb x11-apps mesa-utils
  
# xvfb download
# apt-get update
# apt-get install -y xvfb

FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ARG PYTHON_VERSION=3.10
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL="egl"

# Install dependencies and set up Python in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg

# Install Python + pip + Vulkan utilities
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-pip python3-venv \
    git \
    vulkan-tools \
    libvulkan1 \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libxext6 \
    libjpeg-dev \
    libpng-dev  \
    screen \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Make "python" point to python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Copy everything from the parent directory into container
WORKDIR /workspace
COPY . /workspace

# Upgrade pip and install dependencies
RUN pip install --upgrade pip

# Example for CUDA 12.2 + PyTorch 2.1 (adjust if needed)
# RUN pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0 --extra-index-url https://download.pytorch.org/whl/cu121      
RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# add packaging and ninja
RUN pip install setuptools packaging ninja
RUN pip install flash-attn==2.5.5 --no-build-isolation

# separate this out
RUN cd /workspace/SimplerEnv-OpenVLA/ManiSkill2_real2sim && pip install -e .
RUN cd /workspace/SimplerEnv-OpenVLA && pip install -e .

# install dlip
# RUN git clone https://github.com/kvablack/dlimp
# RUN cd dlimp && pip install -e .

RUN pip install numpy==1.24.4 \
                mediapy \
                tensorflow==2.15.0 \
                regex \
                timm==0.9.10 \
                jax \
                prismatic \
                hydra-core \
                wandb \
                draccus \
                tensorflow_datasets \
                tensorflow_graphics \
                openai \
                sentence_transformers \
                scikit-learn \
                Pillow==9.5.0 \
                gymnasium==0.28.1 \
                ribs==0.8.2 \
                transformers==4.40.1 \
                rich \
                accelerate \
                jsonlines \
                json_numpy \
                git+https://github.com/moojink/dlimp_openvla



CMD ["/bin/bash"]
