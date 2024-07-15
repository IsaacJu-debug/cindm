#FROM nvcr.io/nvidia/pytorch:24.03-py3
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    wget \
    gpg \
    tmux \
    htop \
    libfuse3-dev \
    fuse3 \
    build-essential \
    cmake \
    ffmpeg \
    git \
    python-is-python3 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Configure Microsoft package repository and install blobfuse2
RUN wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb \
    && dpkg -i packages-microsoft-prod.deb \
    && rm packages-microsoft-prod.deb \
    && apt-get update \
    && apt-get install -y blobfuse2 \
    && rm -rf /var/lib/apt/lists/*
# Install azcopy
RUN wget https://aka.ms/downloadazcopy-v10-linux -O /root/azcopy.tar; tar xzf /root/azcopy.tar -C /root; rm /root/azcopy.tar; mv /root/azcopy_linux_amd64_*/azcopy /usr/bin/azcopy; rm -rf /root/azcopy_linux_amd64_*
RUN cd /root; wget https://aka.ms/downloadazcopy-v10-linux; 

# Install Azure CLI
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash; apt clean

# Install our packages
# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Install Python packages
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir --upgrade -r /tmp/requirements.txt \
        --find-links https://data.pyg.org/whl/torch-2.2.1+cu118.html \
        --extra-index-url https://download.pytorch.org/whl/cu118

# Copy and install external packages
COPY ext_packages /tmp/ext_packages
RUN python3 /tmp/ext_packages/install_upstream_python_packages.py