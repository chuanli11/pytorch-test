import time
import torch

list_shapes = [32, 64, 128, 256, 512, 1024]


output = {}

for shape in list_shapes:
    num_warmup = 5
    for i in range(num_warmup):
        rand_tensor = torch.rand((shape, shape, shape))
        tensor = rand_tensor.to("cuda")
        result = tensor * tensor
        torch.cuda.synchronize()

    num_sample = 10
    total_start = time.time()
    for i in range(num_sample):
        start = time.time()
        rand_tensor = torch.rand(shape)
        tensor = rand_tensor.to("cuda")
        result = tensor * tensor
        torch.cuda.synchronize()
        end = time.time()
    total_end = time.time()
    print(
        "Average Inference time for {}: {} sec".format(
            shape, (total_end - total_start) / num_sample
        )
    )
