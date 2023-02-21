import time
import torch

list_shapes = [1000, 2000, 4000]


output = {}

for shape in list_shapes:
    num_sample = 10
    total_start = time.time()
    for i in range(num_sample):
        start = time.time()

        x = torch.randn(shape, shape)
        y = torch.randn(shape, shape)
        z = torch.matmul(x, y)

        torch.cuda.synchronize()

        end = time.time()

        print((end - start) * 1000)
    total_end = time.time()

    print(
        "Total Inference time for {}: {} sec".format(shape, (total_end - total_start))
    )
