import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter




def is_distributed():
    return dist.is_available() and dist.is_initialized()

def setup(rank, world_size):
    if not dist.is_available(): return

    dist.init_process_group(
        backend="nccl", 
        init_method="env://", 
        world_size=world_size, 
        rank=rank
    )
    torch.cuda.set_device(rank)
    print(f"Setting up on rank {rank}")

def cleanup():
    if not is_distributed(): return
    dist.destroy_process_group()

def train(rank, world_size):
    epochs = 1000
    batch_size = 32
    lr=0.001

    is_distributed_run = is_distributed()

    setup(rank, world_size)

    # получаем writer для логирования в тензорборд
    writer = SummaryWriter(log_dir=f"runs/experiment_{rank}") if dist.get_rank() == 0 else None

    # Simple model, loss, optimizer
    model = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Linear(256, 10)
    ).cuda()

    if is_distributed_run:
        model = DDP(model, device_ids=[rank])
    
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Dummy data
    data_loader = [torch.randn(batch_size, 10).cuda() for _ in range(100)]

    for epoch in range(epochs):
        if not is_distributed_run or dist.get_rank() == 0: 
            print(f'epoch: {epoch}')
        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(batch)
            loss = loss_fn(output, batch)

            if is_distributed_run and i % 100 == 0:
                with torch.no_grad():
                    reduced_loss = loss.clone()
                    dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                    reduced_loss /= dist.get_world_size()
            
            if not is_distributed_run or dist.get_rank() == 0: # выводим только с процесса, принадлежащего 0-й видеокарте
                last_loss = loss.item()
                print(f'Loss: {last_loss} ::: step: {i}')
                if is_distributed_run and i % 100 == 0:
                    print(f'Reduced loss: {reduced_loss}')
            
            loss.backward()
            optimizer.step()

    if not is_distributed_run or dist.get_rank() == 0:
        print(f'final loss:    {last_loss}')
    cleanup()

    if writer:
        writer.close()


def main():
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except:
        rank = 0
        world_size = 4
    
    if torch.cuda.is_available():
        train(rank, world_size)
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    main()
    # torchrun --nproc_per_node=4 lesson3_ddp.py 