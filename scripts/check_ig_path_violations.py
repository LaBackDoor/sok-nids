"""Check whether IG straight-line paths violate domain constraints.

Run from the repo root:
    uv run python scripts/check_ig_path_violations.py
"""

import sys
sys.path.insert(0, "experiments/1")

import torch
import numpy as np

from config import ExperimentConfig
from data_loader import load_dataset
from models import load_models
from pa_xai import get_schema

DATASET_SCHEMA_MAP = {
    "nsl-kdd": "NSL-KDD",
    "cic-ids-2017": "CIC-IDS-2017",
    "unsw-nb15": "UNSW-NB15-CICFlowMeter",
    "cse-cic-ids2018": "CSE-CIC-IDS2018",
}

NUM_SAMPLES = 200


def main():
    config = ExperimentConfig()
    device = torch.device("cpu")

    for ds_name in config.ALL_DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        dataset = load_dataset(ds_name, config.data)
        schema = get_schema(DATASET_SCHEMA_MAP[ds_name])

        dnn_model, _, _, _ = load_models(
            config.output_dir, ds_name,
            input_dim=dataset.X_train.shape[1],
            num_classes=dataset.num_classes,
            dnn_config=config.dnn,
            device=device,
        )
        dnn_model.eval()

        from pa_xai.ig import ProtocolAwareIG

        explainer = ProtocolAwareIG(schema, dnn_model, dataset.X_train)

        n = min(NUM_SAMPLES, len(dataset.X_test))
        rng = np.random.default_rng(42)
        indices = rng.choice(len(dataset.X_test), size=n, replace=False)

        violated_count = 0
        max_change_all = 0.0
        all_violated_features = set()

        for i in indices:
            x = dataset.X_test[i].astype(np.float32)
            report = explainer.check_path_violations(x)

            if report["points_with_violations"] > 0:
                violated_count += 1
                max_change_all = max(max_change_all, report["max_abs_change"])
                all_violated_features.update(report["violated_features"])

        print(f"Samples checked:        {n}")
        print(f"Samples with violations: {violated_count}/{n} "
              f"({100*violated_count/n:.1f}%)")
        print(f"Max abs change:          {max_change_all:.6f}")
        if all_violated_features:
            print(f"Violated features:       {sorted(all_violated_features)}")
        else:
            print("No violations — constrain_path=False is safe for this dataset.")


if __name__ == "__main__":
    main()
