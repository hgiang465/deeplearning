import json
import math

import torch

from preprocessing import IMAGENET_MEAN, IMAGENET_STD


def compute_gradcam(model, images, target, target_layer):
    activations = {}

    def save_activations(_module, _inputs, output):
        activations["value"] = output
        output.retain_grad()

    hook = target_layer.register_forward_hook(save_activations)
    try:
        model.zero_grad(set_to_none=True)
        outputs = model(images)
        target(outputs).sum().backward()
        feature_maps = activations["value"]
        gradients = feature_maps.grad
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * feature_maps).sum(dim=1).relu()
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(1),
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False
        ).squeeze(1)
        cam_min = cam.flatten(1).min(dim=1).values[:, None, None]
        cam_max = cam.flatten(1).max(dim=1).values[:, None, None]
        return (cam - cam_min) / (cam_max - cam_min).clamp_min(1e-8)
    finally:
        hook.remove()


def select_gradcam_indices(loader, selection_path, sample_count):
    dataset = loader.dataset
    members = [str(member) for member in dataset.data["member"]]
    member_to_index = {
        member: index for index, member in enumerate(members)
    }
    if selection_path.is_file():
        selected_members = json.loads(
            selection_path.read_text(encoding="utf-8")
        )["members"]
    else:
        selected_indices = torch.randperm(
            len(dataset)
        )[:min(sample_count, len(dataset))].tolist()
        selected_members = [members[index] for index in selected_indices]
        selection_path.write_text(
            json.dumps({"members": selected_members}, indent=2),
            encoding="utf-8"
        )
    missing_members = [
        member for member in selected_members
        if member not in member_to_index
    ]
    if missing_members:
        raise ValueError("Saved Grad-CAM samples are not in the dataset")
    return [member_to_index[member] for member in selected_members]


def save_gradcam_visualization(
    model,
    loader,
    device,
    output_path,
    target_name,
    selected_indices,
    target_layer,
    max_age,
    model_type="scratch"
):
    import matplotlib.pyplot as plt

    if model_type not in ("scratch", "resnet18"):
        raise ValueError("model_type must be either scratch or resnet18.")

    samples = [loader.dataset[index] for index in selected_indices]
    images = torch.stack([sample["image"] for sample in samples]).to(device)
    model.eval()
    with torch.enable_grad():
        outputs = model(images)
        if target_name == "age":
            target = lambda predictions: predictions["age"].squeeze(1)
            titles = [
                f"Age: {(value.item() * max_age):.1f}"
                for value in outputs["age"].squeeze(1).detach().cpu()
            ]
        elif target_name == "gender":
            predicted = outputs["gender"].argmax(dim=1)
            target = lambda predictions: predictions["gender"].gather(
                1, predicted[:, None]
            ).squeeze(1)
            titles = [
                f"Gender: {value.item()}"
                for value in predicted.detach().cpu()
            ]
        else:
            raise ValueError("target_name must be age or gender")
        heatmaps = compute_gradcam(
            model, images, target, target_layer
        ).detach().cpu()

    columns = 4
    rows = math.ceil(len(images) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(12, 3 * rows), squeeze=False
    )
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= len(images):
            continue
        image = images[index].detach().cpu().permute(1, 2, 0)
        if model_type == "resnet18":
            mean = torch.tensor(IMAGENET_MEAN)
            std = torch.tensor(IMAGENET_STD)
            image = image * std + mean
        image = image.clamp(0, 1)
        axis.imshow(image)
        axis.imshow(heatmaps[index], cmap="jet", alpha=0.45)
        axis.set_title(titles[index])
    figure.suptitle(f"Grad-CAM: {target_name}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
