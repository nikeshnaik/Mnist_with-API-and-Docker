FROM ubuntu:18.04

RUN apt-get update && apt-get install python3-pip -y
RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN set -ex && mkdir /repo
WORKDIR /repo

COPY digit_recognizer/ ./digit_recognizer
COPY api/ ./api

ENV LC_ALL C.UTK-8
ENV LANG C.UTF-8
RUN set -ex && pip3 install tensorflow-gpu
RUN set -ex && pip3 install -r api/requirements.txt

EXPOSE 8000
ENV PYTHONPATH /repo
CMD python3 /repo/api/app.py
