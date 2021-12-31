
## Install Qemu to run ARM images on AMD architecture
```
sudo apt-get install qemu binfmt-support qemu-user-static
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
```

## Requirements compile
### AMD
```
docker run --rm -it --workdir /workspace -v $PWD:/workspace python:3.6.9 bash -c "pip3 install pip-tools && pip-compile --generate-hashes requirements.in requirements.cpu.in -o requirements.cpu.txt && pip-compile --generate-hashes requirements.in requirements.cpu.in requirements.dev.in -o requirements.dev.txt"
```
### ARM
```
docker run --rm -it -e LC_ALL=C.UTF-8 -e LANG=C.UTF-8 --workdir /workspace -v $PWD:/workspace nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.9-py3 bash -c "pip3 install pip-tools && pip-compile --generate-hashes requirements.in -o requirements.txt"

docker run --rm -it -e LC_ALL=C.UTF-8 -e LANG=C.UTF-8 --workdir /workspace -v $PWD:/workspace nvcr.io/nvidia/l4t-ml:r32.6.1-py3 bash -c "pip3 install pip-tools && pip-compile --generate-hashes requirements.in -o requirements.txt"
```


## Models
Download from: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md

### Generate PBTXT
TODO - create an automated script

Download and extract model files
```
pip3 install tensorflow==1.12.0
git clone https://github.com/opencv/opencv
cd opencv/samples/dnn
PYTHONPATH=$PWD python3 tf_text_graph_ssd.py \
    --input ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb \
    --config ssd_mobilenet_v2_coco_2018_03_29/pipeline.config \
    --output ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pbtxt
```
