"""Real pretrained encoders for the PILOT. Needs `pip install transformers torchvision` and internet on first use (weights are downloaded).

audio: Wav2Vec 2.0 base (facebook/wav2vec2-base), LAST layer, ~99 frames at 50 Hz for 2 s -> pooled to 50 later.  [D-15 open]
video: ResNet-18 pretrained on ImageNet, one 512-d vector per frame.
       THIS IS A PLACEHOLDER, NOT A LIP-READING ENCODER. The paper's visual-speech encoder is undecided (D-02). A general image
       network sees the face but hardly the lip motion, so detection quality will be poor; say so in the report.
Both return torch tensors with the same interface as features/standins.py, so features/extract.py works unchanged.
"""
import numpy as np
import torch

AUDIO_MODEL = "facebook/wav2vec2-base"


def load_audio_encoder(name: str = AUDIO_MODEL, device: str = "cpu"):
    from transformers import Wav2Vec2Model
    model = Wav2Vec2Model.from_pretrained(name).to(device).eval()

    @torch.no_grad()
    def encode(wav: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.asarray(wav, np.float32))
        x = (x - x.mean()) / (x.std() + 1e-7)                      # same normalisation as the model's feature extractor
        return model(x[None].to(device)).last_hidden_state[0].cpu()   # [~99, 768]
    return encode


def load_video_encoder(device: str = "cpu"):
    import torchvision
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity(); net = net.to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode(frames: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(frames)).to(device).permute(0, 3, 1, 2).float() / 255.0   # [T,3,H,W]
        return net((x - mean) / std).cpu()                                                                 # [T, 512]
    return encode