# 1. Base Image — start from Miniconda (official base image with conda preinstalled)
FROM continuumio/miniconda3

# 2. Working Directory — set where commands will run inside the container
WORKDIR /workspace

# 3. Copy Your Files
COPY envs/ ./envs/
COPY jafar/ ./jafar/
COPY open-genie/ ./open-genie/

# 4. Install Both Environments
RUN conda env create -f envs/genienv.yml \
 && conda env create -f envs/jafarenv.yml

# 5. Set Shell to Use a Specific Conda Environment by Default
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

