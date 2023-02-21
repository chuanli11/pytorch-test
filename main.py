import time
import torch

list_shapes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
num_warmup = 5
num_sample = 10

for shape in list_shapes:
    for i in range(num_warmup):
        x = torch.randn(shape, shape)
        y = torch.randn(shape, shape)
        x = x.to("cuda")
        y = y.to("cuda")
        z = torch.matmul(x, y)
        torch.cuda.synchronize()

    total_time = 0.0
    for i in range(num_sample):
        start = time.time()
        x = torch.randn(shape, shape)
        y = torch.randn(shape, shape)
        x = x.to("cuda")
        y = y.to("cuda")
        z = torch.matmul(x, y)
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
