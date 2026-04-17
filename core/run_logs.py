"""
Structured text logs for Bio-Lattice.

``RunLogWriter`` is the unified logger used for:
- extraction/events append logs (main pipeline),
- per-run training logs,
- per-run metrics logs.
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


def _write_text(path: str, content: str, mode: str = "a") -> None:
    _ensure_parent_dir(path)
    with open(path, mode, encoding="utf-8") as f:
        if content and not content.endswith("\n"):
            f.write(content + "\n")
        else:
            f.write(content)
        f.flush()


class RunLogWriter:
    """Per-run text logger for training or metrics logs."""

    def __init__(self, path: str):
        self.path = path

    @classmethod
    def new_run_path(cls, kind: str = "training") -> str:
        os.makedirs(config.PATH_TRAINING_LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        prefix = "metrics_run" if kind == "metrics" else "training_run"
        return os.path.join(config.PATH_TRAINING_LOG_DIR, f"{prefix}_{ts}.txt")

    @classmethod
    def create_new(cls, kind: str = "training") -> "RunLogWriter":
        return cls(cls.new_run_path(kind=kind))

    def write(self, content: str, mode: str = "a") -> None:
        _write_text(self.path, content, mode=mode)

    def event(self, message: str) -> None:
        """Append one line (useful for extraction/event logs)."""
        _write_text(self.path, message, mode="a")

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

    def write_training_header(
        self,
        *,
        start_iso: str,
        device_str: str,
        n_dataset_train: int,
        n_dataset_val: int,
        excluded_ids_audit: list[str],
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
            f"patients_excluded_by_audit ({len(excluded_ids_audit) if excluded_ids_audit else 0}): {', '.join(excluded_ids_audit) if excluded_ids_audit else 'none'}",
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

    def write_training_footer(
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

    def write_metrics(
        self,
        *,
        total: int,
        accuracy: float,
        auc: float,
        sensitivity: float,
        specificity: float,
        confusion: dict,
        configured_threshold: float,
        best_threshold_youden: dict,
    ) -> None:
        lines = [
            "=== Bio-Lattice metrics run ===",
            f"evaluated_at: {datetime.now().isoformat(timespec='seconds')}",
            f"total_validation_patients: {total}",
            "",
            "--- Metrics at configured threshold ---",
            f"configured_threshold: {configured_threshold:.4f}",
            f"accuracy: {accuracy:.4f}",
            f"roc_auc: {auc:.4f}",
            f"sensitivity: {sensitivity:.4f}",
            f"specificity: {specificity:.4f}",
            "",
            "--- Confusion matrix (configured threshold) ---",
            f"tn: {confusion.get('tn', 0)}",
            f"fp: {confusion.get('fp', 0)}",
            f"fn: {confusion.get('fn', 0)}",
            f"tp: {confusion.get('tp', 0)}",
            "",
            "--- Best threshold by Youden J ---",
            f"threshold: {best_threshold_youden.get('threshold', 0.0):.4f}",
            f"youden_j: {best_threshold_youden.get('youden_j', 0.0):.4f}",
            f"accuracy: {best_threshold_youden.get('accuracy', 0.0):.4f}",
            f"sensitivity: {best_threshold_youden.get('sensitivity', 0.0):.4f}",
            f"specificity: {best_threshold_youden.get('specificity', 0.0):.4f}",
        ]
        self.write("\n".join(lines), mode="w")