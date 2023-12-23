import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
from config import config
from pathlib import Path
import numpy as np
import time
import os
from scripts.models.PVS.pvs import ConvNext_LSTM

# create model and load weights
model = ConvNext_LSTM()
model.load_state_dict(torch.load(Path(config.CHECKPOINT_DIR, "ConvLSTM0.pt")))

# create dummy input/output for model conversion
dummy_input = torch.randn(1, 5, 3, 256, 256)
dummy_output = torch.randn(1, 5, 1, 256, 256)

# convert model to onnx and after that to trt engine
torch.onnx.export(model, dummy_input, Path(config.CHECKPOINT_DIR, "onnx_model.onnx"))
os.system("/usr/src/tensorrt/bin/trtexec --onnx=checkpoints/onnx_model.onnx --saveEngine=model_engine.trt --explicitBatch")

# load trt engine and create runtime
f = open(Path(config.CHECKPOINT_DIR, "model_engine.trt"), "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

# create engine and cuda context
engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# allocate memory for input and output
d_input = cuda.mem_alloc(1 * dummy_input.nelement() * dummy_input.element_size())
d_output = cuda.mem_alloc(1 * dummy_output.nelement() * dummy_output.element_size())

bindings = [int(d_input), int(d_output)]

# create cuda stream and dummy IOs to np array
stream = cuda.Stream()
input = np.array(dummy_input)
output = np.array(dummy_output)

# function to predict one batch
def predict(batch): # result gets copied into output
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # syncronize threads
    stream.synchronize()

    return output

# warm up prediction (first run is always slower)
print("Warming up...")

pred = predict(input)

# run <num_iterations> inferences and measure the time
num_iterations = 1000
start = time.time()

for i in range(num_iterations):
    pred = predict(input)

end = time.time()

print(f"Needed {end-start} seconds for {num_iterations} inferences. This corresponds to {num_iterations/(end-start)} FPS.")
