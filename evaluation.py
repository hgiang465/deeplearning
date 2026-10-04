import torch

from preprocessing import AGE_GROUP_NAMES
from training import (
    AGE_LOSS_WEIGHT,
    GENDER_LOSS_WEIGHT,
    MultitaskLoss,
    run_epoch,
    save_json
)


def evaluate_test(
    model,
    test_loader,
    device,
    output_dir
):
    """Evaluate a loaded best checkpoint once on the untouched test split."""
    metrics, confusion = run_epoch(
        model=model,
        loader=test_loader,
        criterion=MultitaskLoss(
            age_weight=AGE_LOSS_WEIGHT,
            gender_weight=GENDER_LOSS_WEIGHT,
            # weights come from the train split (set in build_loaders)
            group_weights=getattr(test_loader.dataset, "group_weights", None)
        ),
        device=device,
        optimizer=None,
        return_confusion=True
    )
    save_json(
        {
            "metrics": metrics,
            # rows = true age group, columns = predicted age group
            "age_group_names": AGE_GROUP_NAMES,
            "age_group_confusion_matrix": confusion,
            "age_loss_weight": AGE_LOSS_WEIGHT,
            "gender_loss_weight": GENDER_LOSS_WEIGHT
        },
        output_dir / "test_metrics.json"
    )
    return metrics


def load_best_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint