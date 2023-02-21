import time
import torch

list_shapes = [1000, 2000, 4000]


output = {}

for shape in list_shapes:
    num_warmup = 5
    num_sample = 10

    for i in range(num_warmup):
        x = torch.randn(shape, shape)
        y = torch.randn(shape, shape)
        z = torch.matmul(x, y)

    total_start = time.time()
    for i in range(num_sample):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        x = torch.randn(shape, shape)
        y = torch.randn(shape, shape)
        z = torch.matmul(x, y)

        end.record()

        torch.cuda.synchronize()
        print(start.elapsed_time(end))
    total_end = time.time()

    print(
        "Total Inference time for {}: {} sec".format(shape, (total_end - total_start))
    )
