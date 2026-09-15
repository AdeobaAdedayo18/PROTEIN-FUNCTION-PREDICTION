"""Phase 6d: Training loop for the hierarchical GO classifier.

The model operates on variable-length sets of publication embeddings.
Each protein retains all available PMID embeddings. Padding is performed
dynamically within each mini-batch by collate_protein_batch().
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader, random_split

from ldpfp.models.hierarchical_classifier import (
    HierarchicalGOClassifier,
    hierarchical_loss,
)
from ldpfp.training_data import collate_protein_batch


def train(
    dataset,
    parent_child_pairs: list[tuple[int, int]],
    n_labels: int,
    embed_dim: int = 768,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
    lam: float = 0.5,
    val_fraction: float = 0.15,
    device: str | None = None,
    seed: int = 42,
    model_output_path: str | Path | None = None,
):
    """
    Train the hierarchical GO classifier.

    Parameters
    ----------
    dataset:
        ProteinLiteratureDataset. Each sample contains a variable number
        of PMID-level PubMedBERT embeddings and its propagated GO labels.

    parent_child_pairs:
        List of (child_idx, parent_idx) GO relationships used by the
        hierarchical consistency loss.

    n_labels:
        Number of GO terms in the propagated label vocabulary.

    embed_dim:
        Dimension of PMID embeddings. PubMedBERT produces 768-dimensional
        vectors.

    epochs:
        Number of complete training passes.

    batch_size:
        Number of proteins per mini-batch.

    lr:
        AdamW learning rate.

    lam:
        Weight of the hierarchy consistency penalty.

    val_fraction:
        Fraction of proteins reserved for validation.

    device:
        "cuda", "cpu", etc. Automatically selected when None.

    seed:
        Random seed used for reproducible train/validation splitting.

    model_output_path:
        Optional path for saving the final PyTorch checkpoint.

    Returns
    -------
    model:
        Trained HierarchicalGOClassifier.

    history:
        Dictionary containing train and validation losses.
    """

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    print(f"Device: {device}")
    print(f"Dataset proteins: {len(dataset):,}")
    print(f"GO labels: {n_labels:,}")
    print(f"Hierarchy pairs: {len(parent_child_pairs):,}")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")

    if len(dataset) < 2:
        raise ValueError(
            "Dataset must contain at least two proteins "
            "to create train and validation sets."
        )

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generator = torch.Generator().manual_seed(seed)

    # ---------------------------------------------------------
    # Train / validation split
    # ---------------------------------------------------------

    n_val = max(
        1,
        int(round(val_fraction * len(dataset))),
    )

    # Ensure training always receives at least one sample.
    n_val = min(
        n_val,
        len(dataset) - 1,
    )

    n_train = len(dataset) - n_val

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=generator,
    )

    print(f"Training proteins:   {n_train:,}")
    print(f"Validation proteins: {n_val:,}")

    # ---------------------------------------------------------
    # DataLoaders
    #
    # IMPORTANT:
    # Padding occurs inside collate_protein_batch().
    #
    # We therefore never create a global tensor shaped:
    #
    #     (15,579, 114, 768)
    #
    # Each batch is padded only to the largest publication
    # count present in that particular batch.
    # ---------------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_protein_batch,
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_protein_batch,
        pin_memory=device.type == "cuda",
    )

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
    }

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
                "val_fraction": val_fraction,
                "seed": seed,
                "embed_dim": embed_dim,
                "n_labels": n_labels,
                "n_train": n_train,
                "n_val": n_val,
            }
        )

        # =====================================================
        # Epoch loop
        # =====================================================

        for epoch in range(epochs):

            # -------------------------------------------------
            # Training
            # -------------------------------------------------

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
                total_train_loss / n_train
            )

            # -------------------------------------------------
            # Validation
            # -------------------------------------------------

            model.eval()

            total_val_loss = 0.0

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

            val_loss = (
                total_val_loss / n_val
            )

            # -------------------------------------------------
            # Logging
            # -------------------------------------------------

            history["train_loss"].append(
                train_loss
            )

            history["val_loss"].append(
                val_loss
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f}"
            )

        # -----------------------------------------------------
        # Save model
        # -----------------------------------------------------

        if model_output_path is not None:

            model_output_path = Path(
                model_output_path
            )

            model_output_path.parent.mkdir(
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
                },
                model_output_path,
            )

            print(
                f"Model saved → "
                f"{model_output_path}"
            )

        # MLflow's model artifact is useful for experiment
        # tracking independently of the explicit checkpoint.
        mlflow.pytorch.log_model(
            model,
            "model",
        )

    return model, history