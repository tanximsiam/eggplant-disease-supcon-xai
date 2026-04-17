"""SYNTHETIC FIXTURES FOR TESTS ONLY — random-noise images, never used for reported results."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from eggplant.utils import load_config

N_CLASSES = 3


@pytest.fixture
def synthetic_cfg(tmp_path):
    cfg = copy.deepcopy(load_config("configs/experiment.yaml"))
    cache, mdir = tmp_path / "cache", tmp_path / "manifests"
    cache.mkdir(); mdir.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for c in range(N_CLASSES):
        for i in range(6):
            p = cache / f"cls{c}" / f"synthetic_{i}.jpg"
            p.parent.mkdir(exist_ok=True)
            arr = (rng.random((80, 96, 3)) * 80 + c * 60).astype(np.uint8)  # class-dependent brightness
            Image.fromarray(arr).save(p)
            rows.append({"rel_path": f"cls{c}/synthetic_{i}.jpg", "cache_path": f"cls{c}/synthetic_{i}.jpg",
                         "class_id": c, "label_original": f"cls{c}", "group_id": f"g{c}_{i // 2}",
                         "group_type": "synthetic", "sha256": "x"})
    (mdir / "class_map.json").write_text(json.dumps(
        [{"class_id": c, "folder_name": f"cls{c}", "display_name": f"cls{c}"} for c in range(N_CLASSES)]))
    cfg["data"].update(cache_dir=str(cache), manifest_dir=str(mdir))
    cfg["train"].update(image_size=64, batch_size=4, rep_epochs=2, ft_epochs=2, num_workers=0, amp=False)
    return cfg, pd.DataFrame(rows), tmp_path
