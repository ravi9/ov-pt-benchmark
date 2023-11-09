import os
import torch
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
import openvino as ov
import numpy as np
import time
import argparse

def prepare_models(pt_model_fn, ov_model_fn):

  if not os.path.exists(pt_model_fn) or not os.path.exists(ov_model_fn):
    print(f"Preparing PyTorch and OV models...")

    dirname = os.path.dirname(pt_model_fn)
    if not os.path.exists(dirname):
       os.makedirs(dirname)

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    torch.save(model.state_dict(), pt_model_fn)
    print(f"Saved PyTorch model: {pt_model_fn}")

    ov_model = ov.convert_model(model)
    ov.save_model(ov_model, output_model=ov_model_fn)
    print(f"Saved OpenVINO model: {ov_model_fn}")

  else :
    print(f"PyTorch and OV models already exists...using {pt_model_fn} and {ov_model_fn}")

def prepare_input():
    width = 224
    height = 224
    # Generate random pixel values for the image (range: 0 to 255)
    random_image = np.random.randint(0, 256, size=(1, 3, height, width), dtype=np.uint8)
    image = torch.from_numpy(random_image)
    # Normalize the image
    image = image.float() / 255.0
    return image

def benchmark_pt(image, ts_mode=False, precision="f32"):
    model = resnet50()
    model.load_state_dict(torch.load(pt_model_fn))
    model.eval()

    if ts_mode:
        model = torch.jit.script(model)

    if  precision == "bf16" :
        model = model.to(dtype=torch.bfloat16)
        image = image.to(dtype=torch.bfloat16)

    print(f'\nBenchmarking PyTorch Inference w/ TorchScript={ts_mode} with {precision} for {num_iterations} iterations')
    inf_time_arr = []

    # Set the start time
    start_time = time.time()

    if precision == "bf16" :
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            for _ in range(num_iterations):
                output = model(image)
                inf_time = time.time() - start_time
                inf_time_arr.append(inf_time)
    else:
        for _ in range(num_iterations):
            output = model(image)
            inf_time = time.time() - start_time
            inf_time_arr.append(inf_time)

    print(f'PyTorch Average inference time  w/ TorchScript={ts_mode} with {precision}: {np.mean(inf_time_arr):.2f} seconds')

    return inf_time_arr


def benchmark_ov(image, precision):
    core = ov.Core()
    config = {"INFERENCE_PRECISION_HINT": precision}
    target_device = "CPU"
    ov_model = core.read_model(model=ov_model_fn)
    compiled_model = core.compile_model(ov_model, target_device, config)

    print(f'\nBenchmarking OpenVINO Inference for {num_iterations} iterations with {precision}')

    inf_time_arr = []
    # Set the start time
    start_time = time.time()

    for _ in range(num_iterations):
        output = compiled_model(image)
        inf_time = time.time() - start_time
        inf_time_arr.append(inf_time)

    print(f'OpenVINO Average inference time with {precision}: {np.mean(inf_time_arr):.2f} seconds')
    return inf_time_arr

def calculate_statistics(times, num_iterations ):
    return np.sum(times), np.median(times), np.mean(times), np.std(times), np.min(times), np.max(times), num_iterations / np.mean(times)

def print_statistics(metric, values):
    print("{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(metric, *values))

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-pt', '--pt_model_fn', type=str, default='models/resnet50_model.pth')
  parser.add_argument('-ov', '--ov_model_fn', type=str, default='models/resnet50_model.xml')
  parser.add_argument('-pr', '--precision', type=str, choices=['bf16', 'f16', 'f32'], default='bf16')
  parser.add_argument('-ni', '--num_iterations', type=int, default=5000)

  args = parser.parse_args()

  pt_model_fn = args.pt_model_fn
  ov_model_fn = args.ov_model_fn
  precision = args.precision
  num_iterations = args.num_iterations

  prepare_models(pt_model_fn, ov_model_fn)

  image = prepare_input()
  print(f"Image size: {image.shape}")

  pt_fp32_times = benchmark_pt(image)
  pt_ts_fp32_times = benchmark_pt(image, ts_mode=True)
  pt_bf16_times = benchmark_pt(image, precision="bf16")

  ov_fp32_times = benchmark_ov(image, precision="f32")
  ov_bf16_times = benchmark_ov(image, precision="bf16")

  times_dict = {
                "pt_fp32": pt_fp32_times,
                "pt_ts_fp32": pt_ts_fp32_times,
                "pt_bf16": pt_bf16_times,
                "ov_fp32": ov_fp32_times,
                "ov_bf16": ov_bf16_times
                }

  stats_dict = {key: calculate_statistics(times, num_iterations) for key, times in times_dict.items()}

  print(f"\nImage size: {image.shape}")
  print("\nFwk,total_time,median,avg,std_dev,min,max,fps")

  for key, values in stats_dict.items():
    print_statistics(key, values)


#   total_time = [np.sum(pt_fp32_times), np.sum(pt_ts_fp32_times),np.sum(pt_bf16_times), np.sum(ov_fp32_times), np.sum(ov_bf16_times)]
#   median_time = [np.median(pt_fp32_times), np.median(pt_ts_fp32_times),np.median(pt_bf16_times), np.median(ov_fp32_times), np.median(ov_bf16_times)]
#   average_time = [np.mean(pt_fp32_times), np.mean(pt_ts_fp32_times),np.mean(pt_bf16_times), np.mean(ov_fp32_times), np.mean(ov_bf16_times)]
#   std_dev = [np.std(pt_fp32_times), np.std(pt_ts_fp32_times),np.std(pt_bf16_times), np.std(ov_fp32_times), np.std(ov_bf16_times)]
#   min_time = [np.min(pt_fp32_times), np.min(pt_ts_fp32_times),np.min(pt_bf16_times), np.min(ov_fp32_times), np.min(ov_bf16_times)]
#   max_time = [np.max(pt_fp32_times), np.max(pt_ts_fp32_times),np.max(pt_bf16_times), np.max(ov_fp32_times), np.max(ov_bf16_times)]
#   fps = [ num_iterations / np.mean(pt_fp32_times), num_iterations / np.mean(pt_ts_fp32_times), num_iterations / np.mean(pt_bf16_times), num_iterations / np.mean(ov_fp32_times), num_iterations / np.mean(ov_bf16_times)]

#   print(f"\nImage size: {image.shape}")
#   print("\nmetric,pt_fp32,pt_ts_fp32,pt_bf16,ov_fp32,ov_bf16")
#   print("total_time,{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(*total_time))
#   print("median,{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(*median_time))
#   print("average,{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(*average_time))
#   print("std_dev,{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(*std_dev))
#   print("min,{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(*min_time))
#   print("max,{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(*max_time))
#   print("fps,{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(*fps))
