#!/usr/bin/env bash
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

COPY requirements.txt .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt

ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512