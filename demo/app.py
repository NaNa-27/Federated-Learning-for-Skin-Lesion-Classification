"""HAM10000 federated-learning demo (Gradio).

Upload a dermoscopic image and compare the class probabilities of the
Centralized, FedAvg and FedProx models trained in 03_federated_training.ipynb.
For illustration only - this is NOT a diagnostic tool.

Run (from this folder):
    python app.py                      # opens http://127.0.0.1:7860 in the browser
    python app.py --port 7861 --no-browser
Optional: show the ground-truth label when the uploaded file is a HAM10000 image
    python app.py --partition ../federated-learning/output-fl/HAM10000_fl_partition_v1.csv
"""
import argparse
import json
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
import torchvision
from PIL import Image

HERE = Path(__file__).resolve().parent
ARM_LABELS = {"centralized": "Centralized (upper reference)",
              "fedavg": "Federated - FedAvg",
              "fedprox": "Federated - FedProx"}


def build_model(backbone, n_classes):
    if backbone == "efficientnet_b0":
        m = torchvision.models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    elif backbone == "resnet18":
        m = torchvision.models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    else:
        raise ValueError(f"Unknown backbone {backbone}")
    return m


class Predictor:
    def __init__(self, models_dir, device):
        self.card = json.loads((models_dir / "model_card.json").read_text(encoding="utf-8"))
        self.classes = self.card["classes"]
        self.names = self.card["class_names"]
        self.size = self.card["img_size"]
        self.device = device
        self.mean = torch.tensor(self.card["mean"]).view(1, 3, 1, 1)
        self.std = torch.tensor(self.card["std"]).view(1, 3, 1, 1)
        self.models = {}
        for key, info in self.card["models"].items():
            m = build_model(self.card["backbone"], len(self.classes))
            state = torch.load(models_dir / info["file"], map_location="cpu", weights_only=True)
            m.load_state_dict(state)
            self.models[key] = m.eval().to(device)
        print(f"Loaded {list(self.models)} ({self.card['backbone']}) on {device}")

    def preprocess(self, img):
        # Must match training: RGB -> bilinear resize to 224x224 (no crop) -> ImageNet mean/std
        img = img.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
        x = x.view(self.size, self.size, 3).permute(2, 0, 1).unsqueeze(0).float() / 255
        return ((x - self.mean) / self.std).to(self.device)

    @torch.no_grad()
    def __call__(self, img):
        x = self.preprocess(img)
        return {k: m(x).softmax(1)[0].cpu().tolist() for k, m in self.models.items()}

    def label(self, c):
        return f"{c.upper()} - {self.names[c]}"


def make_ui(pred, truth):
    keys = list(pred.models)

    def run(path):
        if path is None:
            return [None] * len(keys) + ["Upload an image to see the predictions."]
        probs = pred(Image.open(path))
        outs = [{pred.label(c): p for c, p in zip(pred.classes, probs[k])} for k in keys]

        rows = ["| Model | Top-1 | Probability |", "|---|---|---|"]
        top = {}
        for k in keys:
            i = max(range(len(pred.classes)), key=lambda j: probs[k][j])
            top[k] = pred.classes[i]
            rows.append(f"| {ARM_LABELS.get(k, k)} | **{pred.label(top[k])}** | {probs[k][i]:.1%} |")
        notes = []
        gt = truth.get(Path(path).stem)
        if gt:
            hits = [ARM_LABELS.get(k, k).split(" (")[0] for k in keys if top[k] == gt["dx"]]
            notes.append(f"**Ground truth** ({gt['image_id']}, {gt['split']} split, client "
                         f"`{gt['client']}`): **{pred.label(gt['dx'])}**. "
                         f"Correct: {', '.join(hits) if hits else 'none'}.")
            if gt["split"] != "test":
                notes.append("_This image was in the training/validation data; the prediction is not a fair test._")
        notes.append("Models agree." if len(set(top.values())) == 1 else
                     "**Models disagree** - this is where federated training differs from centralized training.")
        return outs + ["\n".join(rows) + "\n\n" + "\n\n".join(notes)]

    metrics = " · ".join(f"{ARM_LABELS.get(k, k).split(' (')[0]}: Macro-F1 {v['test_macro_f1']:.3f}"
                         for k, v in pred.card["models"].items())
    with gr.Blocks(title="HAM10000 FL demo") as demo:
        gr.Markdown(
            "# Skin lesion classification: Federated vs. Centralized\n"
            f"Backbone **{pred.card['backbone']}**, 7 classes, trained on HAM10000 with 4 source-based "
            f"clients (lesion-level split, seed 42). Pooled-test {metrics}.\n\n"
            "> ⚠️ For illustration only. **Not a diagnostic tool**; do not use for medical decisions.")
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="filepath", label="Dermoscopic image", height=320)
                btn = gr.Button("Predict", variant="primary")
                summary = gr.Markdown()
            with gr.Column(scale=2):
                with gr.Row():
                    labels = [gr.Label(num_top_classes=len(pred.classes), label=ARM_LABELS.get(k, k))
                              for k in keys]
        btn.click(run, inputs=img, outputs=labels + [summary])
        img.change(run, inputs=img, outputs=labels + [summary])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=HERE / "demo_models")
    ap.add_argument("--partition", type=Path, default=None,
                    help="HAM10000_fl_partition_v1.csv, to show the true label of HAM10000 images")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    pred = Predictor(a.models, device)
    truth = {}
    if a.partition:
        import csv
        with open(a.partition, newline="", encoding="utf-8") as f:
            truth = {r["image_id"]: r for r in csv.DictReader(f)}
        print(f"Ground truth loaded for {len(truth):,} images")
    make_ui(pred, truth).launch(server_name="127.0.0.1", server_port=a.port, inbrowser=not a.no_browser)


if __name__ == "__main__":
    main()
