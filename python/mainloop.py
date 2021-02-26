## mode: python ##
## #coding: utf-8 ##

import os
import torch
import torch.optim as optim
import torch.nn.functional as F

logdirname='log/MNIST'

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(logdirname)

def train(rank, args, model, device, dataset, dataloader_kwargs):
    torch.manual_seed(args.seed + rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    for epoch in range(args.epoch):
        train_epoch(rank, epoch, args, model, device, dataloader, optimizer)


def train_epoch(rank, epoch, args, model, device, data_loader, optimizer):
    model.train()
    pid = os.getpid()

    for x, z in data_loader:
        optimizer.zero_grad()
        loss = model.loss_function(x, z)
        loss.backward()
        optimizer.step()

    if rank % args.num_processes == 0:
        writer.add_scalar("training/loss", loss.item(), epoch)

    print('pid:%d, epoch:%03d, loss: %.8f' % (pid, epoch, loss.item()))


def test(args, model, device, testset):
    torch.manual_seed(args.seed)

    testloader = torch.utils.data.DataLoader(testset, batch_size=2000, shuffle=False)

    data, labels = testloader.__iter__().next()

    # feature
    model.eval()
    with torch.no_grad():
        features = model(data)
        writer.add_embedding(features, metadata=labels)

    writer.close()

# import pdb; pdb.set_trace()

### Local Variables: ###
### truncate-lines:t ###
### End: ###
