ARG PYTORCH="2.1.2"
ARG CUDA="12.1"
ARG CUDNN="8"
ARG DEBIAN_FRONTEND=noninteractive

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="8.6 8.9"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"
ENV TZ="Etc/UTC"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends git ninja-build build-essential \
    libglib2.0-0 libsm6 libxrender-dev libxext6 ffmpeg tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install "numpy<2" && \
    pip install openmim && \
    mim install "mmengine>=0.10.4" "mmcv>=2.1.0"

WORKDIR /mmaction2
COPY . /mmaction2

RUN pip install --no-cache-dir cython && \
    pip install --no-cache-dir -e .
