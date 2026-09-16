"""Phase 6d: Training loop for the hierarchical GO classifier.

The model operates on variable-length sets of publication embeddings.
Each protein retains all available PMID embeddings. Padding is performed
dynamically within each mini-batch by collate_protein_batch().

The dataset is split reproducibly into train, validation, and held-out
test partitions. The test partition is never used during training or
model selection and is reserved for Phase 7 evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Subset

from ldpfp.evaluate import fmax_score
from ldpfp.models.hierarchical_classifier import (
    HierarchicalGOClassifier,
    hierarchical_loss,
)
from ldpfp.training_data import collate_protein_batch


def _get_protein_id(dataset, index: int) -> str:
    """Extract a protein ID from a dataset sample."""
    sample = dataset[index]

    # ProteinLiteratureDataset currently returns:
    # (protein_id, embeddings, labels)
    return str(sample[0])


def create_dataset_splits(
    dataset,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
):
    """Create deterministic train/validation/test splits."""

    if len(dataset) < 3:
        raise ValueError(
            "Dataset must contain at least three proteins "
            "to create train, validation, and test sets."
        )

    total_fraction = (
        train_fraction
        + val_fraction
        + test_fraction
    )

    if abs(total_fraction - 1.0) > 1e-8:
        raise ValueError(
            "train_fraction + val_fraction + "
            "test_fraction must equal 1.0."
        )

    n_total = len(dataset)

    n_train = int(
        round(train_fraction * n_total)
    )

    n_val = int(
        round(val_fraction * n_total)
    )

    # Let test absorb rounding differences.
    n_test = (
        n_total
        - n_train
        - n_val
    )

    if min(n_train, n_val, n_test) < 1:
        raise ValueError(
            "Each dataset split must contain "
            "at least one protein."
        )

    generator = torch.Generator().manual_seed(
        seed
    )

    permutation = torch.randperm(
        n_total,
        generator=generator,
    ).tolist()

    train_indices = permutation[:n_train]

    val_indices = permutation[
        n_train:n_train + n_val
    ]

    test_indices = permutation[
        n_train + n_val:
    ]

    train_ds = Subset(
        dataset,
        train_indices,
    )

    val_ds = Subset(
        dataset,
        val_indices,
    )

    test_ds = Subset(
        dataset,
        test_indices,
    )

    return {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }


def save_dataset_splits(
    dataset,
    splits: dict,
    output_path: str | Path,
    seed: int,
):
    """Persist split membership for reproducible Phase 7 evaluation."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    def protein_ids(indices):
        return [
            _get_protein_id(dataset, i)
            for i in indices
        ]

    split_data = {
        "seed": seed,
        "train": protein_ids(
            splits["train_indices"]
        ),
        "validation": protein_ids(
            splits["val_indices"]
        ),
        "test": protein_ids(
            splits["test_indices"]
        ),
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            split_data,
            f,
            indent=2,
        )

    print(
        f"Dataset splits saved → "
        f"{output_path}"
    )


def train(
    dataset,
    parent_child_pairs: list[tuple[int, int]],
    n_labels: int,
    embed_dim: int = 768,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
    lam: float = 0.5,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    device: str | None = None,
    seed: int = 42,
    model_output_path: str | Path | None = None,
    split_output_path: str | Path | None = None,
    patience: int = 5,
    min_delta: float = 1e-4,
):
    """
    Train the hierarchical GO classifier.

    Model selection is based on validation Fmax.

    The held-out test set is created here for reproducibility but is
    intentionally never loaded or evaluated during training.

    Parameters
    ----------
    dataset:
        ProteinLiteratureDataset containing variable-length PMID
        embedding sets and propagated GO labels.

    parent_child_pairs:
        List of (child_idx, parent_idx) GO hierarchy relationships.

    n_labels:
        Number of GO terms in the propagated vocabulary.

    embed_dim:
        Dimension of PMID embeddings.

    epochs:
        Maximum number of training epochs.

    batch_size:
        Number of proteins per mini-batch.

    lr:
        AdamW learning rate.

    lam:
        Weight of the hierarchy-consistency penalty.

    train_fraction:
        Fraction of proteins assigned to training.

    val_fraction:
        Fraction assigned to validation.

    test_fraction:
        Fraction reserved for final Phase 7 evaluation.

    device:
        Training device. Automatically selected when None.

    seed:
        Random seed used for reproducibility.

    model_output_path:
        Path where the best checkpoint will be saved.

    split_output_path:
        Path where train/validation/test protein IDs are saved.

    patience:
        Number of consecutive epochs without meaningful validation
        Fmax improvement before early stopping.

    min_delta:
        Minimum validation Fmax increase required to count as an
        improvement.

    Returns
    -------
    model:
        Model containing the best validation-Fmax weights.

    history:
        Dictionary containing training and validation metrics.
    """

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(device)

    print(f"Device: {device}")
    print(f"Dataset proteins: {len(dataset):,}")
    print(f"GO labels: {n_labels:,}")
    print(
        f"Hierarchy pairs: "
        f"{len(parent_child_pairs):,}"
    )
    print(
        f"Embedding dimension: "
        f"{embed_dim}"
    )
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Patience: {patience}")
    print(f"Min delta: {min_delta}")

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if patience < 1:
        raise ValueError(
            "patience must be at least 1."
        )

    if min_delta < 0:
        raise ValueError(
            "min_delta cannot be negative."
        )

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---------------------------------------------------------
    # Train / validation / test split
    # ---------------------------------------------------------

    splits = create_dataset_splits(
        dataset=dataset,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    train_ds = splits["train"]
    val_ds = splits["val"]
    test_ds = splits["test"]

    n_train = len(train_ds)
    n_val = len(val_ds)
    n_test = len(test_ds)

    print()
    print("Dataset split")
    print("--------------------------------")
    print(
        f"Training proteins:   "
        f"{n_train:,}"
    )
    print(
        f"Validation proteins: "
        f"{n_val:,}"
    )
    print(
        f"Test proteins:       "
        f"{n_test:,}"
    )
    print("--------------------------------")
    print(
        "Test set is reserved for "
        "Phase 7 evaluation."
    )

    # ---------------------------------------------------------
    # Save split membership
    # ---------------------------------------------------------

    if split_output_path is not None:
        save_dataset_splits(
            dataset=dataset,
            splits=splits,
            output_path=split_output_path,
            seed=seed,
        )

    # ---------------------------------------------------------
    # DataLoaders
    #
    # Padding occurs dynamically within each mini-batch.
    # The full dataset is never padded to the global maximum
    # number of publications.
    # ---------------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_protein_batch,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_protein_batch,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # Deliberately no test_loader.
    #
    # The test split remains untouched until Phase 7.

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = HierarchicalGOClassifier(
        embed_dim=embed_dim,
        n_labels=n_labels,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
    )

    # ---------------------------------------------------------
    # Training history
    # ---------------------------------------------------------

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_fmax": [],
        "val_micro_aupr": [],
        "val_best_threshold": [],
    }

    best_val_fmax = -1.0
    best_epoch = -1
    best_threshold = None

    epochs_without_improvement = 0

    best_state_dict = None

    # ---------------------------------------------------------
    # MLflow
    # ---------------------------------------------------------

    mlflow.set_experiment(
        "ldpfp-hierarchical-classifier"
    )

    with mlflow.start_run():

        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "lambda": lam,
                "train_fraction":
                    train_fraction,
                "val_fraction":
                    val_fraction,
                "test_fraction":
                    test_fraction,
                "seed": seed,
                "embed_dim": embed_dim,
                "n_labels": n_labels,
                "n_train": n_train,
                "n_val": n_val,
                "n_test": n_test,
                "patience": patience,
                "min_delta": min_delta,
                "selection_metric":
                    "validation_fmax",
            }
        )

        # =====================================================
        # Epoch loop
        # =====================================================

        for epoch in range(epochs):

            # =================================================
            # Training
            # =================================================

            model.train()

            total_train_loss = 0.0

            for (
                _protein_ids,
                embeddings,
                mask,
                targets,
            ) in train_loader:

                embeddings = embeddings.to(
                    device,
                    non_blocking=True,
                )

                mask = mask.to(
                    device,
                    non_blocking=True,
                )

                targets = targets.to(
                    device,
                    non_blocking=True,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                logits, _ = model(
                    embeddings,
                    mask,
                )

                loss = hierarchical_loss(
                    logits,
                    targets,
                    parent_child_pairs,
                    lam=lam,
                )

                loss.backward()

                optimizer.step()

                total_train_loss += (
                    loss.item()
                    * embeddings.size(0)
                )

            train_loss = (
                total_train_loss
                / n_train
            )

            # =================================================
            # Validation
            # =================================================

            model.eval()

            total_val_loss = 0.0

            val_targets = []
            val_probabilities = []

            with torch.no_grad():

                for (
                    _protein_ids,
                    embeddings,
                    mask,
                    targets,
                ) in val_loader:

                    embeddings = embeddings.to(
                        device,
                        non_blocking=True,
                    )

                    mask = mask.to(
                        device,
                        non_blocking=True,
                    )

                    targets = targets.to(
                        device,
                        non_blocking=True,
                    )

                    logits, _ = model(
                        embeddings,
                        mask,
                    )

                    loss = hierarchical_loss(
                        logits,
                        targets,
                        parent_child_pairs,
                        lam=lam,
                    )

                    total_val_loss += (
                        loss.item()
                        * embeddings.size(0)
                    )

                    probabilities = torch.sigmoid(
                        logits
                    )

                    val_targets.append(
                        targets.detach().cpu()
                    )

                    val_probabilities.append(
                        probabilities.detach().cpu()
                    )

            val_loss = (
                total_val_loss
                / n_val
            )

            # -------------------------------------------------
            # Combine validation batches
            # -------------------------------------------------

            y_true = torch.cat(
                val_targets,
                dim=0,
            ).numpy()

            y_prob = torch.cat(
                val_probabilities,
                dim=0,
            ).numpy()

            # -------------------------------------------------
            # Validation predictive metrics
            # -------------------------------------------------

            val_fmax, val_best_threshold = (
                fmax_score(
                    y_true,
                    y_prob,
                )
            )

            val_micro_aupr = (
                average_precision_score(
                    y_true,
                    y_prob,
                    average="micro",
                )
            )

            # -------------------------------------------------
            # History
            # -------------------------------------------------

            history["train_loss"].append(
                float(train_loss)
            )

            history["val_loss"].append(
                float(val_loss)
            )

            history["val_fmax"].append(
                float(val_fmax)
            )

            history["val_micro_aupr"].append(
                float(val_micro_aupr)
            )

            history[
                "val_best_threshold"
            ].append(
                float(val_best_threshold)
            )

            # -------------------------------------------------
            # MLflow logging
            # -------------------------------------------------

            mlflow.log_metrics(
                {
                    "train_loss":
                        float(train_loss),

                    "val_loss":
                        float(val_loss),

                    "val_fmax":
                        float(val_fmax),

                    "val_micro_aupr":
                        float(val_micro_aupr),

                    "val_best_threshold":
                        float(
                            val_best_threshold
                        ),
                },
                step=epoch,
            )

            # -------------------------------------------------
            # Epoch output
            # -------------------------------------------------

            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | "
                f"Fmax={val_fmax:.4f} | "
                f"micro-AUPR={val_micro_aupr:.4f} | "
                f"t*={val_best_threshold:.2f}"
            )

            # =================================================
            # Best-model checkpointing
            # =================================================

            improved = (
                val_fmax
                > best_val_fmax + min_delta
            )

            if improved:

                best_val_fmax = float(
                    val_fmax
                )

                best_epoch = epoch + 1

                best_threshold = float(
                    val_best_threshold
                )

                epochs_without_improvement = 0

                # Keep an in-memory copy of the best weights.
                #
                # Moving tensors to CPU ensures this copy does
                # not consume additional GPU memory.
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value
                    in model.state_dict().items()
                }

                if model_output_path is not None:

                    best_path = Path(
                        model_output_path
                    )

                    best_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    torch.save(
                        {
                            "model_state_dict":
                                model.state_dict(),

                            "optimizer_state_dict":
                                optimizer.state_dict(),

                            "n_labels":
                                n_labels,

                            "embed_dim":
                                embed_dim,

                            "parent_child_pairs":
                                parent_child_pairs,

                            "history":
                                history,

                            "seed":
                                seed,

                            "best_epoch":
                                best_epoch,

                            "best_val_fmax":
                                best_val_fmax,

                            "best_threshold":
                                best_threshold,

                            "train_fraction":
                                train_fraction,

                            "val_fraction":
                                val_fraction,

                            "test_fraction":
                                test_fraction,
                        },
                        best_path,
                    )

                print(
                    "  ↳ New best model saved "
                    f"(Fmax={best_val_fmax:.4f})"
                )

            else:

                epochs_without_improvement += 1

                print(
                    "  ↳ No Fmax improvement "
                    f"({epochs_without_improvement}/"
                    f"{patience})"
                )

            # =================================================
            # Early stopping
            # =================================================

            if (
                epochs_without_improvement
                >= patience
            ):

                print()
                print(
                    "Early stopping triggered."
                )

                print(
                    f"No validation Fmax improvement "
                    f"greater than {min_delta} for "
                    f"{patience} consecutive epochs."
                )

                break

        # =====================================================
        # Restore best model
        # =====================================================

        if best_state_dict is not None:

            model.load_state_dict(
                best_state_dict
            )

            model.to(device)

        # -----------------------------------------------------
        # MLflow final metadata
        # -----------------------------------------------------

        mlflow.log_metrics(
            {
                "best_val_fmax":
                    best_val_fmax,

                "best_epoch":
                    best_epoch,

                "best_threshold":
                    (
                        best_threshold
                        if best_threshold is not None
                        else 0.0
                    ),
            }
        )

    # =========================================================
    # Training summary
    # =========================================================

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Epochs completed:     "
        f"{len(history['train_loss'])}"
    )

    print(
        f"Best epoch:           "
        f"{best_epoch}"
    )

    print(
        f"Best validation Fmax: "
        f"{best_val_fmax:.6f}"
    )

    if best_threshold is not None:
        print(
            f"Best threshold:       "
            f"{best_threshold:.2f}"
        )

    if model_output_path is not None:
        print(
            f"Best model:           "
            f"{model_output_path}"
        )

    print(
        "Held-out test set:    "
        "untouched"
    )

    print("=" * 60)

    return model, history