import _init_path
import torch
import feature
from models.conv import GatedConv
import torch.nn.functional as F
from utils.decoder import BeamDecoder

model = GatedConv.load("pretrained/gated-conv.pth")
model.eval()

decoder = BeamDecoder(model.vocabulary,"lm/zh_giga.no_cna_cmn.prune01244.klm")

def predict(f):
    wav = feature.load_audio(f)
    spec = feature.spectrogram(wav)
    spec.unsqueeze_(0)
    with torch.no_grad():
        y = model.cnn(spec)
        y = F.softmax(y, 1)
    y_len = torch.tensor([y.size(-1)])
    y = y.permute(0, 2, 1)  # B * T * V
    print("decoding")
    return decoder.decode(y, y_len)
