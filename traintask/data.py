
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append("..")
from utils.feature import load_audio, spectrogram

class MASRDataset(Dataset):
    def __init__(self, index_path, labels_path):
        with open(index_path,'r', encoding='utf-8') as f:
            idx = f.readlines()
        idx = [x.strip().split(",", 1) for x in idx]
        self.idx = idx
        with open(labels_path, encoding='utf-8') as f:
            labels = json.load(f)
        self.labels = dict([(labels[i], i) for i in range(len(labels))])
        self.labels_str = labels

    def __getitem__(self, index):
        wav, transcript = self.idx[index]
        wav = load_audio(wav)
        spect = spectrogram(wav)
        transcript = list(filter(None, [self.labels.get(x,1) for x in transcript]))

        return spect, transcript

    def __len__(self):
        return len(self.idx)


def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, freq_size, max_seqlength)
    input_lens = torch.IntTensor(minibatch_size)
    target_lens = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x].narrow(1, 0, seq_length).copy_(tensor)
        input_lens[x] = seq_length
        target_lens[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_lens, target_lens


class MASRDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(MASRDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

