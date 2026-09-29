from pathlib import Path
import numpy as np
import torch
from torchvision.models import resnet18
from .imaging import canonical, letterbox, handcrafted

class FeatureExtractor:
    def __init__(self, weights, threads=4):
        torch.set_num_threads(threads)
        self.model = resnet18(weights=None)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model.fc = torch.nn.Identity()
        self.model.eval()

    def extract(self, images, metadata):
        h, g, details, tensors = [], [], [], []
        for a, m in zip(images, metadata):
            hh, gg, detail = handcrafted(a, m)
            h.append(hh); g.append(gg); details.append(detail)
            box = letterbox(canonical(a, m), 224, m["spacing"])
            t = torch.from_numpy(box.copy()).float().div(255).repeat(3,1,1)
            t = (t-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
            tensors.append(t)
        embeddings = []
        with torch.inference_mode():
            for start in range(0,len(tensors),16):
                embeddings.append(self.model(torch.stack(tensors[start:start+16])).numpy())
        return {"hog": np.array(h), "geo": np.array(g), "deep": np.concatenate(embeddings)}, details

def context_features(features, groups, regions):
    """Input-only context: unique peers within the SAME study; no labels or UID features."""
    g = features["geo"]
    out = []
    for i, group in enumerate(groups):
        peers = np.where(np.asarray(groups) == group)[0]
        blocks = []
        for region in ["spine", "hip"]:
            subset = [j for j in peers if regions[j] == region and j != i]
            mean = g[subset].mean(0) if subset else np.zeros(g.shape[1])
            blocks.extend([mean, np.array([len(subset)], dtype=float)])
        out.append(np.concatenate(blocks))
    return np.array(out, np.float32)
