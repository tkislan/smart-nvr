FROM gitpod/workspace-full

RUN pyenv install 3.6.9

RUN apt update -y && \
    apt install -y pkgconf libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev
