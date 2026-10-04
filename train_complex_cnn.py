import argparse
from pathlib import Path

import torch

from evaluation import evaluate_test, load_best_checkpoint
from explainability import (
    save_gradcam_visualization,
    select_gradcam_indices
)
from models.complex_cnn import ComplexCNN
from preprocessing import SEED, build_loaders
from training import MAX_AGE, train_model


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "models" / "complex_cnn"
# Dùng chung danh sách ảnh Grad-CAM với SimpleCNN để so sánh công bằng
GRADCAM_SELECTION_PATH = (
    PROJECT_ROOT / "data" / "models" / "simple_cnn" / "gradcam_samples.json"
)
IMAGE_SIZE = 224
BATCH_SIZE = 32
MAX_EPOCHS = 30
LEARNING_RATE = 1e-3   # train from scratch, không fine-tune pretrained nên dùng LR như SimpleCNN
WEIGHT_DECAY = 1e-4
PATIENCE = 5
GRADCAM_SAMPLE_COUNT = 8
# ComplexCNN train from scratch 
# preprocessing kiểu "scratch" (pixel scaling [0,1]) giống SimpleCNN, KHÔNG
MODEL_TYPE = "scratch"


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def close_datasets(loaders):
    for loader in loaders.values():
        close_method = getattr(loader.dataset, "close", None)
        if callable(close_method):
            close_method()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device()
    loaders = build_loaders(
        model_type=MODEL_TYPE,
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
        seed=SEED
    )
    model = ComplexCNN().to(device)
    try:
        _, checkpoint_path = train_model(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            device=device,
            output_dir=OUTPUT_DIR,
            model_name="ComplexCNN",
            max_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience
        )
        load_best_checkpoint(model, checkpoint_path, device)
        test_metrics = evaluate_test(
            model,
            loaders["test"],
            device,
            OUTPUT_DIR
        )
        print("Test metrics:")
        for name, value in test_metrics.items():
            print(f"{name}: {value:.4f}")

        GRADCAM_SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
        gradcam_indices = select_gradcam_indices(
            loader=loaders["val"],
            selection_path=GRADCAM_SELECTION_PATH,
            sample_count=GRADCAM_SAMPLE_COUNT,
            seed=SEED
        )
        for target_name in ["age", "gender"]:
            save_gradcam_visualization(
                model=model,
                loader=loaders["val"],
                device=device,
                output_path=OUTPUT_DIR / f"gradcam_{target_name}.png",
                target_name=target_name,
                selected_indices=gradcam_indices,
                target_layer=model.get_gradcam_layer(target_name),
                max_age=MAX_AGE,
                model_type=MODEL_TYPE
            )
    finally:
        close_datasets(loaders)


if __name__ == "__main__":
    main()