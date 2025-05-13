import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    model = torch.nn.Linear(10, 1).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    x = torch.randn(16, 10).to(rank)
    y = torch.randn(16, 1).to(rank)

    for epoch in range(3):
        outputs = ddp_model(x)
        loss = loss_fn(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"[GPU {rank}] Epoch {epoch} Loss: {loss.item()}")

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
    # torchrun --nproc_per_node=2 ddp_train.py