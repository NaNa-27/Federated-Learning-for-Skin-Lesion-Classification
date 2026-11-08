# HAM10000 FL demo (Gradio)

Compares the Centralized, FedAvg and FedProx models (EfficientNet-B0) on one uploaded
dermoscopic image. For illustration only, not a diagnostic tool.

## Setup (once)

1. Copy the `demo_models/` folder from `ham10000_fl_results.zip` into this folder:
   `demo/demo_models/{centralized,fedavg,fedprox}_efficientnet_b0.pt` + `model_card.json`.
   The `.pt` files are not committed (see `.gitignore`).
2. From the repo root, in the VS Code terminal:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r demo/requirements.txt
```

A CPU-only PyTorch is enough for the demo.

## Run

- VS Code: Run and Debug (Ctrl+Shift+D) -> **Demo: HAM10000 FL (Gradio)** -> F5, or
- terminal: `python demo/app.py --partition federated-learning/output-fl/HAM10000_fl_partition_v1.csv`

The browser opens http://127.0.0.1:7860. `--partition` is optional: with it, uploading a
HAM10000 image (file name `ISIC_xxxxxxx.jpg`) also shows its true label, split and client.
Use test-split images for a fair demo. Other options: `--port 7861`, `--no-browser`, `--cpu`.
