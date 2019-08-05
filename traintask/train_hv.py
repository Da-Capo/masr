from tqdm import tqdm
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorboardX as tensorboard
import data
import sys
sys.path.append("..")
from utils.decoder import GreedyDecoder
from models.conv import GatedConv
import horovod.torch as hvd

device = "cuda"

def train(
    model,
    epochs=1000,
    batch_size=64,
    train_index_path="../data_aishell/train-sort.manifest",
    dev_index_path="../data_aishell/dev.manifest",
    labels_path="../data_aishell/labels.json",
    learning_rate=0.6,
    momentum=0.8,
    max_grad_norm=0.2,
    weight_decay=0,
):
    hvd.init()
    torch.manual_seed(1024)
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(1024)

    # dataset loader
    train_dataset = data.MASRDataset(train_index_path, labels_path)
    batchs = (len(train_dataset) + batch_size - 1) // batch_size

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_dataloader = data.MASRDataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, sampler=train_sampler
    )


    dev_dataset = data.MASRDataset(dev_index_path, labels_path)
    dev_sampler = torch.utils.data.distributed.DistributedSampler(
        dev_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    dev_dataloader = data.MASRDataLoader(
        dev_dataset, batch_size=batch_size, num_workers=1, sampler=dev_sampler
    )

    # optimizer
    parameters = model.parameters()
    optimizer = torch.optim.SGD(
        parameters,
        lr=learning_rate * hvd.size(),
        momentum=momentum,
        nesterov=True,
        weight_decay=weight_decay,
    )

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                        named_parameters=model.named_parameters(),
                                        compression=compression)


    ctcloss = nn.CTCLoss()
    
    # lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.985)
    writer = tensorboard.SummaryWriter()
    gstep = 0
    for epoch in range(epochs):
        epoch_loss = 0
        # lr_sched.step()
        lr = get_lr(optimizer)
        if hvd.rank() == 0:
            writer.add_scalar("lr/epoch", lr, epoch)
        for i, (x, y, x_lens, y_lens) in enumerate(train_dataloader):
            x = x.to(device)
            out, out_lens = model(x, x_lens)
            out = out.transpose(0, 1).transpose(0, 2).log_softmax(2)
            loss = ctcloss(out, y, out_lens, y_lens)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()

            if hvd.rank() == 0:
                writer.add_scalar("loss/step", loss.item(), gstep)
                gstep += 1
                print(
                    "[{}/{}][{}/{}]\tLoss = {}".format(
                        epoch + 1, epochs, i, int(batchs), loss.item()
                    )
                )
        
        epoch_loss = epoch_loss / batchs
        cer = eval(model, dev_dataloader)
        writer.add_scalar("loss/epoch", epoch_loss, epoch)
        writer.add_scalar("cer/epoch", cer, epoch)
        print("Epoch {}: Loss= {}, CER = {}".format(epoch, epoch_loss, cer))
        torch.save(model.state_dict(), "pretrained/model_{}.pth".format(epoch))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

def eval(model, dataloader):
    model.eval()
    decoder = GreedyDecoder(dataloader.dataset.labels_str)
    cer = 0
    refs = 0
    print("decoding")
    with torch.no_grad():
        for i, (x, y, x_lens, y_lens) in tqdm(enumerate(dataloader)):
            x = x.to(device)
            outs, out_lens = model(x, x_lens)
            outs = F.softmax(outs, 1)
            outs = outs.transpose(1, 2)
            ys = []
            offset = 0
            for y_len in y_lens:
                ys.append(y[offset : offset + y_len])
                offset += y_len
            out_strings, out_offsets = decoder.decode(outs, out_lens)
            y_strings = decoder.convert_to_strings(ys)
            for pred, truth in zip(out_strings, y_strings):
                trans, ref = pred[0], truth[0]
                cer += decoder.cer(trans, ref) 
                refs += ref
        cer /= float(len(refs))
        cer = metric_average(cer, 'cer')
    model.train()
    return cer


if __name__ == "__main__":
    with open("../data_aishell/labels.json") as f:
        vocabulary = json.load(f)
    model = GatedConv(vocabulary)
    model.to(device)
    train(model)
