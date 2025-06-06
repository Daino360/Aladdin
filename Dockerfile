# 1. Base Image - Use Miniconda with CUDA support
FROM continuumio/miniconda3

# 2. Working Directory
WORKDIR /workspace

# 3. Copy Your Files
COPY envs/ ./envs/
COPY jafar/ ./jafar/
COPY open-genie/ ./open-genie/

# 4. First install CUDA toolkit and cuDNN at system level
RUN conda install -y -c "nvidia/label/cuda-12.1.0" \
    cuda-toolkit=12.1 \
    cudnn=8.9.2.26 \
    && conda clean -afy

# 5. Create environments with explicit channel priority
RUN conda env create -f envs/genienv.yml \
    && conda env create -f envs/jafarenv.yml \
    && conda clean -afy

# 6. Set up the default command to use your environment
SHELL ["conda", "run", "-n", "genienv", "/bin/bash", "-c"]