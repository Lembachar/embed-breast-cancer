"""Training throughput and peak VRAM per model and image size, on the dry-run images."""
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from embed_common import Paths
from pipeline_config import available_images, load_metadata
from s3_train import Mammo, build_model

if __name__ == "__main__":
    combos = [((768, 512), 6), ((512, 384), 12)]
    p = Paths(True)
    d = available_images(load_metadata(p, True), p)
    d["y"] = 0
    d = pd.concat([d] * 3)
    torch.backends.cudnn.benchmark = True
    for size, bs in combos:
        for arch in ["resnet50", "efficientnet_v2_s", "convnext_tiny"]:
            dl = DataLoader(Mammo(d, p.png, size, "y", True), batch_size=bs, shuffle=True, num_workers=3, pin_memory=True, persistent_workers=True)
            m, _ = build_model(arch, 2)
            m.cuda().train()
            o = torch.optim.AdamW(m.parameters(), 1e-4)
            sc = torch.amp.GradScaler("cuda")
            c = nn.CrossEntropyLoss()
            torch.cuda.reset_peak_memory_stats()
            it = iter(dl)
            n, t0 = 0, time.time()
            for i in range(40):
                x, y, _i = next(it)
                if i == 8:
                    t0, n = time.time(), 0
                with torch.amp.autocast("cuda"):
                    l = c(m(x.cuda(non_blocking=True)), y.cuda())
                sc.scale(l).backward()
                sc.step(o)
                sc.update()
                o.zero_grad(set_to_none=True)
                n += len(x)
            print(arch, size, "bs", bs, f"{n / (time.time() - t0):.1f} img/s, peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
            del m, o, it, dl
            torch.cuda.empty_cache()
    sys.exit(0)
