"""
Structured text logs for Bio-Lattice: extraction (main pipeline) and per-run training.

Use ``ExtractionLogWriter`` in ``main.py`` (append-only extraction_log.txt) and
``TrainingRunLogWriter`` in ``train.py`` (one file per run under training_logs/).
"""
from __future__ import annotations

import math
import os
import platform
import sys
from datetime import datetime

import config


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


class ExtractionLogWriter:
    """Append-only log for DICOM extraction / registration events (``PATH_LOGS``)."""

    def __init__(self, log_path: str | None = None):
        self.path = log_path or config.PATH_LOGS

    def event(self, message: str) -> None:
        _ensure_parent_dir(self.path)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")
            f.flush()


class TrainingRunLogWriter:
    """One training run file: header, epoch lines, notes, footer."""

    def __init__(self, path: str):
        self.path = path

    @classmethod
    def new_run_path(cls) -> str:
        os.makedirs(config.PATH_TRAINING_LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return os.path.join(config.PATH_TRAINING_LOG_DIR, f"training_run_{ts}.txt")

    @classmethod
    def create_new(cls) -> TrainingRunLogWriter:
        return cls(cls.new_run_path())

    def write(self, content: str, mode: str = "a") -> None:
        _ensure_parent_dir(self.path)
        with open(self.path, mode, encoding="utf-8") as f:
            if content and not content.endswith("\n"):
                f.write(content + "\n")
            else:
                f.write(content)
            f.flush()

    @staticmethod
    def _host_training_context_lines(device_str: str) -> list[str]:
        import torch

        lines = [
            f"platform: {platform.platform()}",
            f"machine: {platform.machine()}",
            f"processor: {platform.processor() or 'n/a'}",
            f"python: {sys.version.split()[0]}",
            f"python_executable: {sys.executable}",
            f"cpus_logical: {os.cpu_count()}",
            f"torch_version: {torch.__version__}",
            f"training_device: {device_str}",
        ]
        if torch.cuda.is_available():
            try:
                lines.append(f"cuda_device_0: {torch.cuda.get_device_name(0)}")
            except Exception:
                lines.append("cuda_device_0: (unavailable)")
        return lines

    def write_run_header(
        self,
        *,
        start_iso: str,
        device_str: str,
        n_dataset_train: int,
        n_dataset_val: int,
        train_size: int,
        val_size: int,
        train_batch_size: int,
        train_drop_last: bool,
        train_batches_per_epoch: int,
        val_batches_per_epoch: int,
        class_balance_lines: tuple[str, ...],
    ) -> None:
        lines = [
            "=== Bio-Lattice training run ===",
            f"started_at: {start_iso}",
            "",
            "--- Host / runtime ---",
            *self._host_training_context_lines(device_str),
            "",
            "--- Dataset (after clinical + cube filters) ---",
            f"dataset_train_rows: {n_dataset_train}",
            f"dataset_val_rows: {n_dataset_val}",
            f"split_fraction: {config.TRAIN_VAL_SPLIT_FRACTION}",
            f"train_split_size: {train_size}",
            f"val_split_size: {val_size}",
            f"train_batch_size: {train_batch_size}",
            f"train_drop_last: {train_drop_last}",
            f"train_batches_per_epoch: {train_batches_per_epoch}",
            f"val_batches_per_epoch: {val_batches_per_epoch}",
            "",
        ]
        if class_balance_lines:
            lines.extend(["--- Train/val label counts (binary Mol Subtype) ---", *class_balance_lines, ""])
        lines.extend(
            [
                "--- Config snapshot ---",
                f"BATCH_SIZE (config): {config.BATCH_SIZE}",
                f"EPOCHS (max): {config.EPOCHS}",
                f"LEARNING_RATE: {config.LEARNING_RATE}",
                f"ONECYCLE_MAX_LR: {config.ONECYCLE_MAX_LR}",
                f"EARLY_STOPPING_PATIENCE: {config.EARLY_STOPPING_PATIENCE}",
                f"EARLY_STOPPING_MIN_DELTA: {config.EARLY_STOPPING_MIN_DELTA}",
                f"EARLY_STOPPING_START_EPOCH: {config.EARLY_STOPPING_START_EPOCH}",
                f"TRAIN_FILTER_BY_AUDIT: {bool(getattr(config, 'TRAIN_FILTER_BY_AUDIT', False))}",
                f"TRAIN_STRATIFIED_SPLIT: {getattr(config, 'TRAIN_STRATIFIED_SPLIT', False)}",
                f"TRAIN_USE_WEIGHTED_SAMPLER: {getattr(config, 'TRAIN_USE_WEIGHTED_SAMPLER', False)}",
                f"TRAIN_USE_CLASS_POS_WEIGHT: {getattr(config, 'TRAIN_USE_CLASS_POS_WEIGHT', False)}",
                f"PATH_MICRO_CUBES: {config.PATH_MICRO_CUBES}",
                f"PATH_MODEL_WEIGHTS: {config.PATH_MODEL_WEIGHTS}",
                "",
                "--- Status ---",
                "training_loop: pending",
                "",
                "--- Epoch log (one line per epoch) ---",
                "",
            ]
        )
        self.write("\n".join(lines), mode="w")

    @staticmethod
    def format_epoch_line(
        *,
        epoch: int,
        epochs_max: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        saved_best: bool,
        best_val_loss: float,
        epochs_without_improve: int,
    ) -> str:
        val_str = f"{val_loss:.6f}" if not math.isnan(val_loss) else "nan"
        return (
            f"epoch {epoch}/{epochs_max} | train_loss: {train_loss:.6f} | val_loss: {val_str} | "
            f"lr: {lr:.8f} | saved_best: {saved_best} | best_val_loss: {best_val_loss:.6f} | "
            f"epochs_no_improve: {epochs_without_improve}"
        )

    def append_epoch_line(
        self,
        *,
        epoch: int,
        epochs_max: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        saved_best: bool,
        best_val_loss: float,
        epochs_without_improve: int,
    ) -> None:
        line = self.format_epoch_line(
            epoch=epoch,
            epochs_max=epochs_max,
            train_loss=train_loss,
            val_loss=val_loss,
            lr=lr,
            saved_best=saved_best,
            best_val_loss=best_val_loss,
            epochs_without_improve=epochs_without_improve,
        )
        self.write(line, mode="a")

    def append_note(self, message: str) -> None:
        text = message if message.startswith("note:") else f"note: {message}"
        self.write(text, mode="a")

    def write_run_footer(
        self,
        *,
        end_iso: str,
        duration_sec: float,
        epochs_completed: int,
        best_val_loss: float,
        early_stopped: bool,
        status: str,
        extra: str = "",
    ) -> None:
        block = [
            "",
            "=== Training finished ===",
            f"ended_at: {end_iso}",
            f"duration_seconds: {duration_sec:.1f}",
            f"epochs_completed: {epochs_completed}",
            f"best_val_loss: {best_val_loss}",
            f"early_stopped: {early_stopped}",
            f"status: {status}",
        ]
        if extra:
            block.append(f"note: {extra}")
        self.write("\n".join(block), mode="a")
