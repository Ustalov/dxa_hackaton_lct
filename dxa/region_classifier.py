"""Original region_cls: preserve full field and aspect, top-left letterbox 224."""
from pathlib import Path
from functools import lru_cache
import json
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18
from PIL import Image

WEIGHT = Path('/data/dxa-ct-synth/model/region_cls/region_cls.pt')
INFO = WEIGHT.with_name('info.json')
CLASSES = ['spine', 'hip_L', 'hip_R']


def prepare(raw):
    h, w = raw.shape
    scale = 224 / max(h, w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    image = np.asarray(Image.fromarray(raw.astype('uint8')).resize(size, Image.Resampling.BILINEAR))
    padded = np.zeros((224, 224), np.uint8)
    padded[:image.shape[0], :image.shape[1]] = image
    return torch.from_numpy(padded).float()[None, None].div(255).sub(.45).div(.25).repeat(1, 3, 1, 1)


class RegionClassifier:
    def __init__(self, weights=WEIGHT):
        info = json.loads(INFO.read_text())
        if info['classes'] != CLASSES:
            raise ValueError('Unexpected region classifier class order')
        torch.set_num_threads(4)
        self.net = resnet18(weights=None)
        self.net.fc = nn.Linear(512, 3)
        self.net.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        self.net.eval()

    @torch.inference_mode()
    def predict(self, raw):
        scores = self.net(prepare(raw)).softmax(1)[0].numpy()
        name = CLASSES[int(scores.argmax())]
        return dict(class_name=name, region='spine' if name == 'spine' else 'hip',
                    side='' if name == 'spine' else 'left' if name == 'hip_L' else 'right',
                    score=float(scores.max()), scores={k:float(v) for k,v in zip(CLASSES,scores)},
                    model=str(WEIGHT), preprocessing='224x224 top-left letterbox, full field, original aspect',
                    score_calibrated=False)


@lru_cache(maxsize=1)
def get_region_classifier():
    return RegionClassifier()
