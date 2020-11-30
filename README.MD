# TensoRT-Solo


<a href="https://996.icu"><img src="https://img.shields.io/badge/link-996.icu-red.svg" alt="996.icu" /></a>


TensorRT for Solo


## Test Enviroments

    ubuntu 18.04 
    jetpack 4.4
    CUDA 10.0
    TensorRT7.1
    pytorch1.4

## Usage

### A quick demo

### Compile 

    mkdir build 
    cd build
    cmake ..
    make -j8

### Convert solo model form pytorch to onnx

    cd ..
    python3 script/get_model_gn.py --config ${SOLO}/configs/solov2_r101_fpn_8gpu_3x.py --checkpoint SOLOv2_R101_3x.pth --outputname solo_r101.onnx 

### Generate engine

    cd build
    ./cuda_engine -i ../solo_r101.onnx -o solo_r101_fp16.engine

### Test

    ./infer_gpupost -e solo_r101_fp16.engine -i ../data/demo.jpg -show -save

## inference performance

Model | Mode | GPU | inference time | Ap
--- |:---:|:---:|:---:|:---:
R101 | FP16 | V100 | 35ms | -
R101 | FP16 | xavier  | 150ms | -
