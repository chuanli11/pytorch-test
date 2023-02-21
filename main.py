import time
import torch

list_shapes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
# list_shapes = [1024]
num_warmup = 5
num_sample = 10
USE_CUDA = True


def test_fn(shape, use_cuda):
    x = torch.randn(shape, shape)
    y = torch.randn(shape, shape)
    if use_cuda:
        x = x.to("cuda")
        y = y.to("cuda")
    z = torch.matmul(x, y)


for shape in list_shapes:
    for i in range(num_warmup):
        test_fn(shape, USE_CUDA)
        torch.cuda.synchronize()

    total_time = 0.0
    for i in range(num_sample):
        start = time.time()
        test_fn(shape, USE_CUDA)
        torch.cuda.synchronize()
        end = time.time()
        ms = (end - start) * 1000
        print("Inference time: {:.4f} ms".format(ms))
        total_time += ms
    print(
        "Average Inference time for {}: {:.4f} ms".format(
            shape, (total_time) / num_sample
        )
    )
