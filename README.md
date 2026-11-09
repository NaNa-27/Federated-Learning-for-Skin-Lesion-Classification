# Federated Learning for Skin Lesion Classification

A codebase for evaluating centralized and federated skin-lesion classification on HAM10000, with a small Gradio demo for comparing trained models.

## Repository structure

```text
├── data-understanding/
│   ├── 01_ham10000_lesion_split.ipynb
│   └── output-d.u/
├── dataset/
│   ├── HAM10000_metadata.csv
|   ├── ISIC2018_Task3_Training_LesionGroupings.csv
|   └── ISIC2018_Task3_Training_GroundTruth/
├── federated-learning/
│   ├── 02_federated_client_partition.ipynb
│   ├── federated-learning-dpl.ipynb
│   ├── fedbn-extension.ipynb
│   ├── output-fl/
│   ├── ham10000_fl_results/
│   └── ham10000_fedbn_results/
└── demo/
    ├── app.py
    ├── requirements.txt
    └── demo_models/
```

## Code and generated outputs

| Component | Purpose | Main outputs |
|---|---|---|
| `data-understanding/01_ham10000_lesion_split.ipynb` | Creates and validates the lesion-level train/validation/test split. | Split manifest, split summary, class counts, validation checks and split visualization in `data-understanding/output-d.u/`. |
| `federated-learning/02_federated_client_partition.ipynb` | Assigns images to four clients using HAM10000 acquisition-source metadata. | Client partition CSV, client/class summaries, cross-source lesion checks and distribution visualization in `federated-learning/output-fl/`. |
| `federated-learning/federated-learning-dpl.ipynb` and `ham10000_fl_results/fl_train.py` | Runs the centralized, local-only, FedAvg and FedProx experiments. | Training logs, per-run metrics and histories, predictions, aggregated result tables, figures and demo checkpoints in `federated-learning/ham10000_fl_results/`. |
| `federated-learning/fedbn-extension.ipynb` and `ham10000_fedbn_results/fl_train.py` | Runs the supplementary FedBN experiment. | Per-seed run records, logs, pooled and per-client evaluation tables in `federated-learning/ham10000_fedbn_results/`. |

The main experiment results are summarized in:

- `federated-learning/ham10000_fl_results/results/main_results.csv`
- `federated-learning/ham10000_fl_results/results/robustness_backbone.csv`
- `federated-learning/ham10000_fedbn_results/results_fedbn/fedbn_A_pooled_test.csv`

## Demo

The Gradio app compares predictions from the **Centralized**, **FedAvg** and **FedProx** EfficientNet-B0 checkpoints included in `demo/demo_models/`.

From the repository root:

```bash
pip install -r demo/requirements.txt
python demo/app.py
```

The app opens at `http://127.0.0.1:7860`. To also display the ground-truth label, split and client for an uploaded HAM10000 image, provide the partition file:

```bash
python demo/app.py --partition federated-learning/output-fl/HAM10000_fl_partition_v1.csv
```

Add `--cpu` to force CPU inference. This demo is for model illustration only and is **not a diagnostic tool**.

## Data note

The repository includes HAM10000 metadata and project manifests/results, but not the original dermoscopic image files. The images must be obtained separately for full training or evaluation. The included demo checkpoints can be used to run the demo without retraining.

## License

The repository code is released under the Apache License 2.0. Third-party dataset files remain subject to their respective attribution and license terms.
