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

device = "cuda"

def train(
    model,
    epochs=100,
    start_epoch=0,
    batch_size=48,
    train_index_path="../data_aishell/train-sort.manifest",
    dev_index_path="../data_aishell/dev.manifest",
    labels_path="../data_aishell/labels.json",
    learning_rate=0.06,
    momentum=0.8,
    max_grad_norm=0.2,
    weight_decay=0,
):
    train_dataset = data.MASRDataset(train_index_path, labels_path)
    batchs = (len(train_dataset) + batch_size - 1) // batch_size
    dev_dataset = data.MASRDataset(dev_index_path, labels_path)
    train_dataloader = data.MASRDataLoader(
        train_dataset, batch_size=batch_size, num_workers=1
    )
    train_dataloader_shuffle = data.MASRDataLoader(
        train_dataset, batch_size=batch_size, num_workers=1, shuffle=True
    )
    dev_dataloader = data.MASRDataLoader(
        dev_dataset, batch_size=batch_size, num_workers=1
    )
    parameters = model.parameters()
    optimizer = torch.optim.SGD(
        parameters,
        lr=learning_rate,
        momentum=momentum,
        nesterov=True,
        weight_decay=weight_decay,
    )
    ctcloss = nn.CTCLoss()
    # lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.985)
    writer = tensorboard.SummaryWriter()
    gstep = 0
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        if epoch > 0:
            train_dataloader = train_dataloader_shuffle
        # lr_sched.step()
        lr = get_lr(optimizer)
        writer.add_scalar("lr/epoch", lr, epoch)
        for i, (x, y, x_lens, y_lens) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            out, out_lens = model(x, x_lens)
            out = out.transpose(0, 1).transpose(0, 2).log_softmax(2)
            loss = ctcloss(out, y, out_lens, y_lens)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()
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


def eval(model, dataloader):
    model.eval()
    decoder = GreedyDecoder(dataloader.dataset.labels_str)
    cer = 0
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
                cer += decoder.cer(trans, ref) / float(len(ref))
        cer /= len(dataloader.dataset)
    model.train()
    return cer


if __name__ == "__main__":
    model = GatedConv(json.load(open("../data_aishell/labels.json", encoding='utf-8')))
    epoch = 40
    model.load_state_dict(torch.load("pretrained/model_{}.pth".format(epoch)))
    print("reload model: pretrained/model_{}.pth".format(epoch))
    model.to(device)
    train(model, start_epoch=epoch)
