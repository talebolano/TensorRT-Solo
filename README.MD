# TensoRT-Solo

The project hosting is used to implement the SOLO algorithm based on TensorRT.

## Test Enviroments

    ubuntu 18.04 
    jetpack 4.4
    CUDA 10.0
    TensorRT7.1

## Usage

### A quick demo

### Compile 

    mkdir build 
    cd build
    cmake ..
    make -j8

### Convert solo model form pytorch to onnx

    cd ..
    python3 script/get_model_gn.py --config ${SOLO pwd}/configs/solov2_r101_fpn_8gpu_3x.py --checkpoint SOLOv2_R101_3x.pth --outputname solo_r101.onnx 

### Generate engine

    cd build
    ./cuda_engine -i ../solo_r101_onnx -o solo_r101_fp16.engine

### test

    ./infer_gpupost -e solo_r101_fp16.engine -i data/demo.jpg -show

## inference performance

Model | Mode | GPU | inference time | Ap
--- |:---:|:---:|:---:|:---:
R101 | FP16 | V100 | 35ms | -
R101 | FP16 | xavier  | 150ms | -
