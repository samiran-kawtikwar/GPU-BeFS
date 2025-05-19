FROM registry.codeocean.com/codeocean/ubuntu:20.04-cuda11.7.0-cudnn8

ARG DEBIAN_FRONTEND=noninteractive

# Install wget, gcc-11, and g++-11
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common wget && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y --no-install-recommends gcc-11 g++-11 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 && \
    apt-get purge -y software-properties-common && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Install minimal CUDA components (compiler + runtime)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-compiler-12-9 \
        cuda-cudart-dev-12-9 && \
    rm -rf /var/lib/apt/lists/*