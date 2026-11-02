"""评估任意模型文件（内部使用，供 _bench.py 调用）。

用法: python scripts/_eval_model.py outputs/models/xxx.pt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from patchcore_qc.config import OUTPUT_ROOT, load_config
from patchcore_qc.dataset import build_test_items
from patchcore_qc.metrics import evaluate_predictions
from patchcore_qc.patchcore import PatchCore


def main(path: str) -> None:
    model = PatchCore.load(path, device="auto")
    cfg = load_config()
    items = build_test_items(cfg)
    res = model.predict_items(items, verbose=False)
    cal_path = Path(path + ".calibration.json")
    thr = json.loads(cal_path.read_text(encoding="utf-8"))["image_score"] if cal_path.exists() else 0.0
    met = evaluate_predictions(items, res["scores"], res["maps"][:, 0], threshold=thr)
    out = {"model": Path(path).name, "grid": model.grid_size, "coreset": int(model.memory.features.shape[0]),
           "image_auroc": met["image_auroc"], "pixel_auroc": met["pixel_auroc"], "aupro": met["aupro"],
           "aupro_0_3": met["aupro_0_3"], "f1": met["f1"], "precision": met["precision"], "recall": met["recall"],
           "threshold": thr,
           "per_class": {k: v["image_auroc"] for k, v in met["per_defect_auroc"].items()}}
    print("RESULT " + json.dumps(out, ensure_ascii=False))
    dest = OUTPUT_ROOT / "runs" / f"bench_{Path(path).stem}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({**out, "metrics": met}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
