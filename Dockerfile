###
# Created on September 05, 2018
#
# author: mae-ma
# attention: Dockerfile to setup a full gym install
# contact: albus.marcel@gmail.com (Marcel Albus)
# version: 1.0.0
#
# #############################################################################################
#
# History:
# - v1.0.0: first init
###
FROM ubuntu:16.04

# To enable direct python print output on shell
# or use "python -u" option
ENV PYTHONUNBUFFERED=0
ENV DEBIAN_FRONTEND=noninteractive

###################
# Gym
###################
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update -q \
    && apt-get install -y \
    software-properties-common

RUN DEBIAN_FRONTEND=noninteractive \
    add-apt-repository --yes ppa:deadsnakes/ppa \
    && apt-get update -q

RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update -q \
    && apt-get install -y \
    python3.6 \
    python3.6-dev \
    python3-pip \
    python-setuptools \
    python3-tk \
    wget \
    unzip \
    git

RUN pip3 install --upgrade pip

WORKDIR /
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt

RUN rm /usr/bin/python
RUN ln -s /env/bin/python3.6 /usr/bin/python
RUN ln -s /env/bin/pip3.6 /usr/bin/pip
RUN ln -s /env/bin/pytest /usr/bin/pytest


###################
# Nvidia Cuda
###################
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates apt-transport-https gnupg-curl && \
    rm -rf /var/lib/apt/lists/* && \
    NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
    apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub && \
    echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list




###################
# MuJoCo Phyiscs engine
###################
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update -q \
    && apt-get install -y \
    curl \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    xpra \
    xserver-xorg-dev


# Replace 1000 with your user / group id
RUN export uid=1000 gid=1000 && \
    mkdir -p /home/developer && \
    echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
    echo "developer:x:${uid}:" >> /etc/group && \
    mkdir -p /etc/sudoers.d/ && \
    echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
    chmod 0440 /etc/sudoers.d/developer && \
    chown ${uid}:${gid} -R /home/developer


#USER developer
ENV HOME /home/developer

###################
# Clean up
###################
RUN apt-get install -y firefox

RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*


COPY /lab1 $HOME
# CMD ["python3", "-u", "Lab1-Problem3.py"]
# CMD ["python3", "Lab1-Problem3.py"]

USER developer
COPY getid_linux $HOME

# CMD ["/usr/bin/firefox"]
