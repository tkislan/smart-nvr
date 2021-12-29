
## Install Qemu to run ARM images on AMD architecture
```
sudo apt-get install qemu binfmt-support qemu-user-static
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
```

## Requirements compile
### AMD
```
docker run --rm -it --workdir /workspace -v $PWD:/workspace python:3.6.9 bash -c "pip3 install pip-tools && pip-compile --generate-hashes requirements.cpu.in -o requirements.cpu.txt && pip-compile --generate-hashes requirements.cpu.in requirements.dev.in -o requirements.dev.txt"
```
### ARM
```
docker run --rm -it -e LC_ALL=C.UTF-8 -e LANG=C.UTF-8 --workdir /workspace -v $PWD:/workspace nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.9-py3 bash -c "pip3 install pip-tools && pip-compile --generate-hashes requirements.in -o requirements.txt"
```
