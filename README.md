# ov-pt-benchmark

# Setup

```bash
sudo apt-get update
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# restart terminal
conda create --name ov-pt-bench python=3.9 -y
conda activate ov-pt-bench
pip install openvino-dev[pytorch]

mkdir ov-pt-bench
cd ov-pt-bench
python bench_resnet50_pt_ov.py
```

Sample output:
```bash
PyTorch and OV models already exists...using models/resnet50_model.pth and models/resnet50_model.xml
Image size: torch.Size([1, 3, 224, 224])

Benchmarking PyTorch Inference w/ TorchScript=False for 5000 iterations
PyTorch Average inference time  w/ TorchScript=False with f32: 32.04 seconds

Benchmarking PyTorch Inference w/ TorchScript=True for 5000 iterations
PyTorch Average inference time  w/ TorchScript=True with f32: 29.69 seconds

Benchmarking OpenVINO Inference for 5000 iterations with f32
OpenVINO Average inference time with f32: 11.30 seconds

Benchmarking OpenVINO Inference for 5000 iterations with bf16
OpenVINO Average inference time with bf16: 4.43 seconds

Image size: torch.Size([1, 3, 224, 224])

metric,pt_fp32,pt_ts_fp32,ov_fp32,ov_bf16
total_time,160220.10,148436.02,56496.71,22154.57
median,32.04,29.70,11.29,4.43
average,32.04,29.69,11.30,4.43
std_dev,18.48,16.99,6.48,2.54
min,0.03,0.10,0.03,0.02
max,64.07,59.10,22.51,8.83
fps,0.03,0.03,0.09,0.23
```
