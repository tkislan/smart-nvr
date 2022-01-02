#!/bin/bash -e

SCRIPT_RELATIVE_DIR=$(dirname "${0}")
ROOT_DIR=$(cd "${SCRIPT_RELATIVE_DIR}/.."; pwd -P)

MODELS="ssd_mobilenet_v2_coco_2018_03_29 ssdlite_mobilenet_v2_coco_2018_05_09"

opencv_tmpdir="$(mktemp -d)"
mkdir -p "${opencv_tmpdir}"
echo -n "Cloning OpenCV ... "
git clone -q --depth 1 https://github.com/opencv/opencv "${opencv_tmpdir}"
echo "done"

for model in ${MODELS}; do
    model_dir="${ROOT_DIR}/data/models/${model}"

    if test -d "${model_dir}"; then
        echo "Model ${model} already exists"
        continue
    fi

    tmpdir="$(mktemp -d)"
    mkdir -p "${tmpdir}"

    pushd "${tmpdir}"

    model_url="http://download.tensorflow.org/models/object_detection/${model}.tar.gz"
    echo -n "Downloading model ${model} from ${model_url} ... "
    wget -q "${model_url}"
    echo "done"

    tar -xf "${model}.tar.gz"

    PYTHONPATH="${opencv_tmpdir}/samples/dnn" \
        python "${opencv_tmpdir}/samples/dnn/tf_text_graph_ssd.py" \
            --input "${model}/frozen_inference_graph.pb" \
            --config "${model}/pipeline.config" \
            --output "${model}/frozen_inference_graph.pbtxt"
    
    mkdir -p "${model_dir}"
    cp -v "${model}/frozen_inference_graph.pb" \
        "${model}/pipeline.config" \
        "${model}/frozen_inference_graph.pbtxt" \
        "${model_dir}/"

    popd

    rm -rv "${tmpdir}"
done

rm -r "${opencv_tmpdir}"
