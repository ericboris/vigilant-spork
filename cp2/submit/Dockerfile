FROM python:3
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# You should install any dependencies you need here.

# These should be installed by python already
#import os
#import random
#import time, math

# These need to be installed in Docker
#import torch
#import torch.nn as nn
#from torch.autograd import Variable
#import pickle

#from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

FROM pytorch/pytorch:latest

RUN apt-get update \
     && apt-get install -y \
        libgl1-mesa-glx \
        libx11-xcb1 \
     && apt-get clean all \
     && rm -r /var/lib/apt/lists/*

RUN /opt/conda/bin/conda install --yes \
    astropy \
    matplotlib \
    pandas \
    scikit-learn \
    scikit-image

RUN pip install torch
RUN pip install pickle-mixin
