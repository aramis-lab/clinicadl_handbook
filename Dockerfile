FROM registry.gitlab.inria.fr/sed-paris/mpp/containers/mpp-python-minimal:latest-gpu


RUN apt-get update -yq \
    && apt-get install make -yq \
    && apt-get clean -y

RUN conda install poetry

RUN cd clinicadl_handbook
RUN make env.dev
