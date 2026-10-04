import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from preprocessing import AGE_GROUP_EDGES, AGE_GROUP_NAMES, NUM_AGE_GROUPS


MAX_AGE = 116.0
AGE_LOSS_WEIGHT = 10.0
GENDER_LOSS_WEIGHT = 1.0
# Huber threshold in years: errors below it are squared, above it are linear
AGE_HUBER_YEARS = 5.0
_GROUP_EDGES = torch.tensor(AGE_GROUP_EDGES, dtype=torch.float32)


class MultitaskLoss(nn.Module):
    def __init__(
        self,
        max_age=MAX_AGE,
        age_weight=AGE_LOSS_WEIGHT,
        gender_weight=GENDER_LOSS_WEIGHT,
        group_weights=None,
        huber_years=AGE_HUBER_YEARS
    ):
        super().__init__()
        self.max_age = max_age
        self.age_weight = age_weight
        self.gender_weight = gender_weight
        # SmoothL1 on age/MAX_AGE; beta is converted from years to that scale
        self.age_criterion = nn.SmoothL1Loss(
            reduction="none", beta=huber_years / max_age
        )
        self.gender_criterion = nn.CrossEntropyLoss()
        # Per-age-group weights, computed from the TRAIN split only
        if group_weights is None:
            self.group_weights = None
        else:
            self.register_buffer(
                "group_weights",
                torch.as_tensor(group_weights, dtype=torch.float32)
            )

    def forward(self, outputs, true_age, true_gender, age_groups=None):
        per_sample = self.age_criterion(
            outputs["age"],
            true_age / self.max_age
        )
        if self.group_weights is not None and age_groups is not None:
            weights = self.group_weights.to(age_groups.device)[age_groups]
            per_sample = per_sample * weights.unsqueeze(1)
        age_loss = per_sample.mean()
        gender_loss = self.gender_criterion(
            outputs["gender"],
            true_gender
        )
        total_loss = (
            self.age_weight * age_loss
            + self.gender_weight * gender_loss
        )
        return total_loss, age_loss, gender_loss


class EpochMetrics:
    def __init__(self):
        self.samples = 0
        self.total_loss = 0.0
        self.age_loss = 0.0
        self.gender_loss = 0.0
        self.absolute_age_error = 0.0
        self.squared_age_error = 0.0
        self.correct_gender = 0
        self.group_abs_error = torch.zeros(NUM_AGE_GROUPS, dtype=torch.float64)
        self.group_count = torch.zeros(NUM_AGE_GROUPS, dtype=torch.float64)
        self.confusion = torch.zeros(
            NUM_AGE_GROUPS, NUM_AGE_GROUPS, dtype=torch.int64
        )
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0

    def update(self, batch_size, total_loss, age_loss, gender_loss,
               predicted_age, true_age, predicted_gender, true_gender,
               age_groups):
        self.samples += batch_size
        self.total_loss += total_loss.item() * batch_size
        self.age_loss += age_loss.item() * batch_size
        self.gender_loss += gender_loss.item() * batch_size
        age_error = predicted_age - true_age
        self.absolute_age_error += age_error.abs().sum().item()
        self.squared_age_error += age_error.square().sum().item()
        self.correct_gender += (predicted_gender == true_gender).sum().item()
        true_group = age_groups.detach().cpu()
        predicted_group = torch.bucketize(
            predicted_age.detach().cpu().reshape(-1).float(),
            _GROUP_EDGES,
            right=True
        )
        absolute = age_error.detach().abs().reshape(-1).cpu().double()
        self.group_abs_error.index_add_(0, true_group, absolute)
        self.group_count.index_add_(0, true_group, torch.ones_like(absolute))
        self.confusion += torch.bincount(
            true_group * NUM_AGE_GROUPS + predicted_group,
            minlength=NUM_AGE_GROUPS * NUM_AGE_GROUPS
        ).reshape(NUM_AGE_GROUPS, NUM_AGE_GROUPS)
        self.true_positive += (
            (predicted_gender == 1) & (true_gender == 1)
        ).sum().item()
        self.false_positive += (
            (predicted_gender == 1) & (true_gender == 0)
        ).sum().item()
        self.false_negative += (
            (predicted_gender == 0) & (true_gender == 1)
        ).sum().item()

    def compute(self):
        if self.samples == 0:
            raise RuntimeError("Cannot calculate metrics from an empty loader")
        precision_denominator = self.true_positive + self.false_positive
        recall_denominator = self.true_positive + self.false_negative
        precision = (
            self.true_positive / precision_denominator
            if precision_denominator else 0.0
        )
        recall = (
            self.true_positive / recall_denominator
            if recall_denominator else 0.0
        )
        f1_denominator = precision + recall
        f1 = (
            2.0 * precision * recall / f1_denominator
            if f1_denominator else 0.0
        )
        present = self.group_count > 0
        group_mae = self.group_abs_error / self.group_count.clamp_min(1)
        total = self.confusion.sum().item()
        index = torch.arange(NUM_AGE_GROUPS)
        adjacent = (index[:, None] - index[None, :]).abs() <= 1
        group_metrics = {
            "age_mae_macro": group_mae[present].mean().item(),
            "age_group_accuracy": self.confusion.diag().sum().item() / total,
            "age_group_adjacent_accuracy": (
                self.confusion[adjacent].sum().item() / total
            )
        }
        for position, name in enumerate(AGE_GROUP_NAMES):
            group_metrics[f"age_mae_{name}"] = (
                group_mae[position].item()
                if present[position] else float("nan")
            )
        return {
            "loss": self.total_loss / self.samples,
            # unweighted MSE on age/MAX_AGE (comparable with earlier runs)
            "age_mse_normalized": (
                self.squared_age_error / self.samples / MAX_AGE ** 2
            ),
            # the (weighted SmoothL1) age loss actually used for training
            "age_loss": self.age_loss / self.samples,
            "gender_cross_entropy": self.gender_loss / self.samples,
            "age_mae": self.absolute_age_error / self.samples,
            "age_rmse": math.sqrt(self.squared_age_error / self.samples),
            "gender_accuracy": self.correct_gender / self.samples,
            "gender_precision": precision,
            "gender_recall": recall,
            "gender_f1": f1,
            **group_metrics
        }


def run_epoch(model, loader, criterion, device, optimizer=None,
              return_confusion=False):
    training = optimizer is not None
    model.train(training)
    metrics = EpochMetrics()
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            images = batch["image"].to(device)
            true_age = batch["age"].to(device)
            true_gender = batch["gender"].to(device)
            age_groups = batch["age_group"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            total_loss, age_loss, gender_loss = criterion(
                outputs, true_age, true_gender, age_groups
            )
            if training:
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            predicted_age = (
                outputs["age"] * criterion.max_age
            ).clamp(0.0, criterion.max_age)
            predicted_gender = outputs["gender"].argmax(dim=1)
            metrics.update(
                images.size(0), total_loss, age_loss, gender_loss,
                predicted_age, true_age, predicted_gender, true_gender,
                age_groups
            )
    if return_confusion:
        return metrics.compute(), metrics.confusion.tolist()
    return metrics.compute()


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def save_json(data, output_path):
    Path(output_path).write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )


def save_checkpoint(output_path, model, optimizer, epoch, metrics, config):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_metrics": metrics,
        "config": config
    }, output_path)


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    output_dir,
    model_name,
    max_epochs,
    learning_rate,
    weight_decay,
    patience
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    group_weights = getattr(train_loader.dataset, "group_weights", None)
    criterion = MultitaskLoss(
        age_weight=AGE_LOSS_WEIGHT,
        gender_weight=GENDER_LOSS_WEIGHT,
        group_weights=group_weights
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )
    config = {
        "model": model_name,
        "parameter_count": count_parameters(model),
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "early_stopping_patience": patience,
        "max_age": MAX_AGE,
        "age_loss": (
            f"SmoothL1 (beta={AGE_HUBER_YEARS} years) on age/MAX_AGE, "
            "weighted by train-derived age-group weights"
        ),
        "age_group_weights": (
            None if group_weights is None else group_weights.tolist()
        ),
        "age_loss_weight": AGE_LOSS_WEIGHT,
        "gender_loss": "CrossEntropyLoss",
        "gender_loss_weight": GENDER_LOSS_WEIGHT,
        "checkpoint_metric": "validation total loss"
    }
    checkpoint_path = output_dir / "best_model.pt"
    history = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, max_epochs + 1):
        start_time = time.perf_counter()
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_metrics = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])
        history.append({
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - start_time,
            "train": train_metrics,
            "val": val_metrics
        })
        save_json(history, output_dir / "history.json")
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path, model, optimizer, epoch, val_metrics, config
            )
        else:
            epochs_without_improvement += 1
        print(
            f"{model_name} epoch {epoch:02d}/{max_epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_MAE={val_metrics['age_mae']:.2f} | "
            f"val_macroMAE={val_metrics['age_mae_macro']:.2f} | "
            f"val_grp_acc={val_metrics['age_group_accuracy']:.4f} | "
            f"val_acc={val_metrics['gender_accuracy']:.4f}"
        )
        if epochs_without_improvement >= patience:
            break
    return history, checkpoint_path