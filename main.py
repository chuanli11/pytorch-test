import time
import torch

list_shapes = [32, 64, 128, 256, 512, 1024]


output = {}

for shape in list_shapes:
    num_warmup = 5
    for i in range(num_warmup):
        x = torch.randn(shape, shape)
        y = torch.randn(shape, shape)
        x = x.to("cuda")
        y = y.to("cuda")
        z = torch.matmul(x, y)
        torch.cuda.synchronize()

    num_sample = 10
    total_start = time.time()
    for i in range(num_sample):
        start = time.time()
        x = torch.randn(shape, shape)
        y = torch.randn(shape, shape)
        x = x.to("cuda")
        y = y.to("cuda")
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        end = time.time()
        print((end - start) * 1000)
    total_end = time.time()
    print(
        "Total Inference time for {}: {} sec".format(shape, (total_end - total_start))
    )
