FROM registry.gitlab.inria.fr/sed-paris/mpp/containers/mpp-python-minimal:latest-gpu

USER root 

RUN apt-get update -yq \
    && apt-get install make -yq \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

RUN micromamba install --yes --name base poetry -c conda-forge 
