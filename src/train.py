import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from utils import (
    load_json,
    save_json,
    ensure_dir,
    get_device,
    set_seed,
    AverageMeter
)

from logger import log_metrics

from dataset import (
    load_records,
    build_label_map,
    save_label_map,
    filter_records_by_label_map,
    split_records,
    ShirtDataset
)

from transforms import (
    get_train_transform,
    get_val_transform
)

from model import ShirtEncoder

from losses import supervised_contrastive_loss


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    use_contrastive=False,
    contrastive_weight=0.5,
    temperature=0.07,
    log_every=20,
    epoch=0
):
    model.train()

    loss_meter = AverageMeter()
    ce_meter = AverageMeter()
    contrastive_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch_idx, batch in enumerate(loader):
        images, labels = batch

        images = images.to(device)
        labels = labels.to(device)

        embeddings, logits = model(images)

        loss_ce = criterion(logits, labels)
        loss = loss_ce

        loss_contrastive = torch.tensor(0.0, device=device)

        if use_contrastive:
            loss_contrastive = supervised_contrastive_loss(
                embeddings=embeddings,
                labels=labels,
                temperature=temperature
            )

            loss = loss + contrastive_weight * loss_contrastive

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()

        batch_size = images.size(0)

        loss_meter.update(loss.item(), batch_size)
        ce_meter.update(loss_ce.item(), batch_size)
        contrastive_meter.update(loss_contrastive.item(), batch_size)
        acc_meter.update(acc, batch_size)

        if log_every and batch_idx % log_every == 0:
            print(
                f"[Epoch {epoch}][Batch {batch_idx}] "
                f"loss={loss_meter.avg:.4f} "
                f"ce={ce_meter.avg:.4f} "
                f"contrastive={contrastive_meter.avg:.4f} "
                f"acc={acc_meter.avg:.4f}"
            )

    return loss_meter.avg, acc_meter.avg


def validate(
    model,
    loader,
    criterion,
    device
):
    if loader is None:
        return None, None

    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            _, logits = model(images)

            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()

            batch_size = images.size(0)

            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc, batch_size)

    return loss_meter.avg, acc_meter.avg


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    label_to_id,
    metrics=None
):
    ensure_dir(os.path.dirname(path))

    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "label_to_id": label_to_id,
        "metrics": metrics or {}
    }

    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.json"
    )
    args = parser.parse_args()

    config = load_json(args.config)

    set_seed(42)

    device = get_device(
        config["training"].get("device", "auto")
    )

    print(f"Using device: {device}")

    # -------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------
    train_records_path = config["data"]["train_records"]
    val_records_path = config["data"].get("val_records")

    image_size = config["data"].get("image_size", 128)
    batch_size = config["data"].get("batch_size", 32)
    num_workers = config["data"].get("num_workers", 0)

    epochs = config["training"].get("epochs", 20)
    lr = config["training"].get("learning_rate", 1e-3)
    weight_decay = config["training"].get("weight_decay", 1e-4)

    use_contrastive = config["training"].get("use_contrastive", False)
    contrastive_weight = config["training"].get("contrastive_weight", 0.5)
    temperature = config["training"].get("temperature", 0.07)
    log_every = config["training"].get("log_every", 20)

    checkpoint_dir = config["output"].get("checkpoint_dir", "checkpoints")
    log_path = config["output"].get("log_path", "logs/experiments.jsonl")
    label_map_path = config["output"].get("label_map_path", "outputs/label_map.json")

    best_checkpoint_path = config["output"].get(
        "best_checkpoint",
        os.path.join(checkpoint_dir, "best.pt")
    )

    last_checkpoint_path = config["output"].get(
        "last_checkpoint",
        os.path.join(checkpoint_dir, "last.pt")
    )

    ensure_dir(checkpoint_dir)

    # -------------------------------------------------------------------
    # Load records
    # -------------------------------------------------------------------
    if not os.path.exists(train_records_path):
        raise FileNotFoundError(
            f"Train records not found: {train_records_path}. "
            "Run scripts/make_manifest.py first."
        )

    train_records = load_records(train_records_path)

    if len(train_records) == 0:
        raise ValueError("Train records are empty.")

    val_records = []

    if val_records_path and os.path.exists(val_records_path):
        val_records = load_records(val_records_path)

    if len(val_records) == 0:
        print("No validation records found. Creating validation split from train records.")

        train_records, val_records, _ = split_records(
            train_records,
            val_frac=0.1,
            test_frac=0.0,
            seed=42
        )

    # -------------------------------------------------------------------
    # Label map
    # -------------------------------------------------------------------
    label_to_id = build_label_map(train_records)

    save_label_map(
        label_map=label_to_id,
        path=label_map_path
    )

    train_records = filter_records_by_label_map(
        train_records,
        label_to_id
    )

    val_records = filter_records_by_label_map(
        val_records,
        label_to_id
    )

    num_classes = len(label_to_id)

    print(f"Number of training records: {len(train_records)}")
    print(f"Number of validation records: {len(val_records)}")
    print(f"Number of classes: {num_classes}")

    # -------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------
    train_transform = get_train_transform(image_size=image_size)
    val_transform = get_val_transform(image_size=image_size)

    train_dataset = ShirtDataset(
        records=train_records,
        label_to_id=label_to_id,
        transform=train_transform,
        return_asset_id=False
    )

    val_dataset = ShirtDataset(
        records=val_records,
        label_to_id=label_to_id,
        transform=val_transform,
        return_asset_id=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=use_contrastive
    )

    val_loader = None

    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False
        )

    # -------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------
    model = ShirtEncoder(
        num_classes=num_classes,
        embedding_dim=config["model"].get("embedding_dim", 128),
        backbone_name=config["model"].get("backbone", "resnet18")
    )

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=epochs
    )

    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            use_contrastive=use_contrastive,
            contrastive_weight=contrastive_weight,
            temperature=temperature,
            log_every=log_every,
            epoch=epoch
        )

        val_loss, val_acc = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        metrics = {
            "event": "epoch",
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "lr": current_lr
        }

        if val_loss is not None:
            metrics["val_loss"] = val_loss
            metrics["val_acc"] = val_acc

        log_metrics(
            log_path=log_path,
            metrics=metrics
        )

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f}, "
            f"val_loss={val_loss}, "
            f"val_acc={val_acc}, "
            f"lr={current_lr:.6f}"
        )

        save_checkpoint(
            path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            label_to_id=label_to_id,
            metrics=metrics
        )

        if val_acc is not None:
            if val_acc >= best_val_acc:
                best_val_acc = val_acc

                save_checkpoint(
                    path=best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    label_to_id=label_to_id,
                    metrics=metrics
                )

                print(f"Saved best checkpoint with val_acc={val_acc:.4f}")
        else:
            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                label_to_id=label_to_id,
                metrics=metrics
            )

    print("Training complete.")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Last checkpoint: {last_checkpoint_path}")


if __name__ == "__main__":
    main()