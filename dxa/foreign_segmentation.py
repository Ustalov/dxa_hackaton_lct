"""Full-resolution foreign-object masks. Presence is derived from these masks only."""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from .foreign_objects import input_tensor


class Block(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                         nn.GroupNorm(8, outgoing), nn.SiLU(),
                         nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
                         nn.GroupNorm(8, outgoing), nn.SiLU())


class ForeignSegmentationNet(nn.Module):
    def __init__(self, local_skip=False):
        super().__init__()
        self.encoder = resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        self.decoders = nn.ModuleList([Block(768, 128), Block(256, 64),
                                      Block(128, 32), Block(96, 16)])
        self.local_skip = local_skip
        self.full = Block(19 if local_skip else 16, 8)
        self.head = nn.Conv2d(8, 1, 1)
        nn.init.constant_(self.head.bias, -3.)

    def forward(self, x):
        local = x
        size = x.shape[-2:]
        e = self.encoder
        a = e.relu(e.bn1(e.conv1(x)))
        b = e.layer1(e.maxpool(a))
        c = e.layer2(b)
        d = e.layer3(c)
        z = e.layer4(d)
        for block, skip in zip(self.decoders, [d, c, b, a]):
            z = block(torch.cat([F.interpolate(z, skip.shape[-2:], mode='bilinear',
                                               align_corners=False), skip], dim=1))
        z = F.interpolate(z, size, mode='bilinear', align_corners=False)
        if self.local_skip:
            z = torch.cat([z, local], dim=1)
        return self.head(self.full(z))


@torch.no_grad()
def probability_map(net, image, horizontal_tta=False):
    images = [image, np.fliplr(image).copy()] if horizontal_tta else [image]
    tensors, boxes = zip(*(input_tensor(im) for im in images))
    p = net(torch.stack(tensors).to(next(net.parameters()).device)).float().sigmoid().cpu().numpy()[:, 0]
    maps = []
    for k, (y, x, nh, nw) in enumerate(boxes):
        heat = cv2.resize(p[k, y:y+nh, x:x+nw], (image.shape[1], image.shape[0]))
        maps.append(np.fliplr(heat).copy() if k else heat)
    return np.mean(maps, axis=0)


def components(probability, spacing, pixel_threshold=.5, min_area_mm2=3.):
    """Local component confidence; no image classifier, anatomy veto or case lookup."""
    count, ids, stats, _ = cv2.connectedComponentsWithStats(
        np.uint8(probability >= pixel_threshold), connectivity=8)
    found = []
    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA] * spacing[0] * spacing[1])
        if area < min_area_mm2:
            continue
        values = probability[ids == label]
        found.append(dict(label=label, area_mm2=area, mean=float(values.mean()),
                          top16=float(np.sort(values)[-16:].mean()),
                          bbox_xywh=[int(v) for v in stats[label, :4]]))
    return ids, found


def decide(probability, spacing, policy):
    ids, found = components(probability, spacing, policy['pixel_threshold'], policy['min_area_mm2'])
    for c in found:
        c['accepted'] = c[policy['score_method']] >= policy['presence_threshold']
    accepted = [c['label'] for c in found if c['accepted']]
    return dict(present=bool(accepted), score=max((c[policy['score_method']] for c in found), default=0.),
                components=found), np.isin(ids, accepted).astype('uint8')
