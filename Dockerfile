FROM python:3.6.9


ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/project/app

ENV TZ=Europe/Bratislava

RUN apt update -y && \
    apt install libgl1-mesa-glx -y && \
    apt install -y pkgconf && \
    apt install -y libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev
