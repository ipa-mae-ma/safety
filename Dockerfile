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

WORKDIR /root
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt

RUN rm /usr/bin/python
RUN ln -s /env/bin/python3.6 /usr/bin/python
RUN ln -s /env/bin/pip3.6 /usr/bin/pip
RUN ln -s /env/bin/pytest /usr/bin/pytest



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

###################
# Clean up
###################
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*



COPY /lab1 .
RUN ls
# CMD ["python3", "-u", "Lab1-Problem3.py"]
CMD ["python3", "Lab1-Problem3.py"]
