import os
import math
import time
from datetime import datetime
# Mandatory patch for Mac (Apple Silicon): Allows complex 3D operations like MaxPool3d to run softly
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

import config
from core import helper
from core.audit import RunLogWriter


# --- Model architecture ---
class ResidualBlock3D(nn.Module):
    """3D ResNet-style block."""
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
        )
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        return self.relu(self.shortcut(x) + self.conv(x))


class BioLattice3DResNet(nn.Module):
    """3D-ResNet classifier for micro-cube tensors."""

    def __init__(self):
        super().__init__()
        c = config
        self.prep = nn.Sequential(
            nn.Conv3d(c.INPUT_CHANNELS, c.STEM_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm3d(c.STEM_CHANNELS),
            nn.ReLU(),
        )
        self.block1 = ResidualBlock3D(c.RES_BLOCK1_IN, c.RES_BLOCK1_OUT)
        self.pool1 = nn.Conv3d(
            c.RES_BLOCK1_OUT, c.RES_BLOCK1_OUT, kernel_size=c.POOL_KERNEL, stride=c.POOL_STRIDE
        )

        self.block2 = ResidualBlock3D(c.RES_BLOCK2_IN, c.RES_BLOCK2_OUT)
        self.pool2 = nn.Conv3d(
            c.RES_BLOCK2_OUT, c.RES_BLOCK2_OUT, kernel_size=c.POOL_KERNEL, stride=c.POOL_STRIDE
        )

        self.avgpool_flatten = nn.Sequential(
            nn.AvgPool3d(kernel_size=c.CLASSIFIER_AVG_POOL_KERNEL, stride=c.CLASSIFIER_AVG_POOL_STRIDE),
            nn.Flatten()
        )
        
        # Dynamically compute the flattened size after spatial layers.
        # This makes the model robust to changes in MICRO_CUBE_SIZE, POOL_KERNEL, etc.
        # without requiring manual recalculation of CLASSIFIER_LINEAR_IN.
        with torch.no_grad():
            _dummy = torch.zeros(1, c.INPUT_CHANNELS, c.MICRO_CUBE_SIZE, c.MICRO_CUBE_SIZE, c.MICRO_CUBE_SIZE)
            _dummy = self.pool1(self.block1(self.prep(_dummy)))
            _dummy = self.pool2(self.block2(_dummy))
            _linear_in = self.avgpool_flatten(_dummy).shape[1]
        
        self.meta_mlp = nn.Sequential(
            nn.Linear(c.META_FEATURE_DIM, c.META_MLP_OUT),
            nn.Mish(),
            nn.BatchNorm1d(c.META_MLP_OUT)
        )

        self.classifier = nn.Sequential(
            nn.Linear(_linear_in + c.META_MLP_OUT, c.CLASSIFIER_HIDDEN),
            nn.BatchNorm1d(c.CLASSIFIER_HIDDEN),
            nn.Mish(),
            nn.Dropout(c.CLASSIFIER_DROPOUT),
            nn.Linear(c.CLASSIFIER_HIDDEN, c.CLASSIFIER_HIDDEN // 2),
            nn.BatchNorm1d(c.CLASSIFIER_HIDDEN // 2),
            nn.Mish(),
            nn.Linear(c.CLASSIFIER_HIDDEN // 2, 1),
        )

        # Grad-CAM support
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, meta_features, return_cam=False):
        x = self.prep(x)
        x = self.pool1(self.block1(x))
        
        # Capture activations for Grad-CAM
        x = self.block2(x)
        if return_cam:
            self.activations = x
            # Enable gradient tracking for this intermediate activation
            self.activations.retain_grad()
        
        x = self.pool2(x)
        x_img = self.avgpool_flatten(x)
        m_feat = self.meta_mlp(meta_features)

        x_combined = torch.cat([x_img, m_feat], dim=1)
        return self.classifier(x_combined)

    def get_gradcam_weights(self):
        if self.activations is None or self.activations.grad is None:
            return None
        # Mean intensity of gradients per channel (GAP)
        return torch.mean(self.activations.grad, dim=[2, 3, 4], keepdim=True)


# --- Training utilities ---
class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance."""
    def __init__(self, alpha=0.75, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)

    def forward(self, inputs, targets):
        pw = getattr(self, "pos_weight", None)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none", pos_weight=pw
        )
        pt = torch.exp(-bce_loss)
        return (self.alpha * ((1 - pt) ** self.gamma) * bce_loss).mean()


class BioLatticeTrainer:
    """Encapsulates training logic, validation, and logging."""

    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, run_log=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.run_log = run_log

        self.best_val_loss = float('inf')
        self.epochs_without_improve = 0

    @property
    def log_path(self):
        return self.run_log.path if self.run_log else None

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for inputs, metas, labels in self.train_loader:
            inputs, metas, labels = inputs.to(self.device), metas.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs, metas)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            running_loss += loss.item()
        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        if len(self.val_loader) == 0:
            return float("nan")
        for inputs, metas, labels in self.val_loader:
            inputs, metas, labels = inputs.to(self.device), metas.to(self.device), labels.to(self.device)
            outputs = self.model(inputs, metas)
            loss = self.criterion(outputs, labels)
            val_loss += loss.item()
        return val_loss / len(self.val_loader)

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def log_to_file(self, content, mode="a"):
        if self.run_log:
            self.run_log.write(content, mode=mode)


def train_model():
    """High-level orchestration for training."""
    t_start_setup = time.monotonic()
    allowed_ids, excluded_ids = helper.load_allowed_patient_ids_from_audits()
    device = helper.get_device()
    print(f"Computing device: {device}")

    # 1. Dataset & Split
    dataset_train = helper.BioLatticeDataset(config.PATH_CLINICAL, config.PATH_MICRO_CUBES, augment=True, allowed_patient_ids=allowed_ids)
    dataset_val = helper.BioLatticeDataset(config.PATH_CLINICAL, config.PATH_MICRO_CUBES, augment=False, allowed_patient_ids=allowed_ids)
    
    split = config.TRAIN_VAL_SPLIT_FRACTION
    train_ds, val_ds, train_size, val_size = helper.build_train_val_subsets(
        dataset_train, dataset_val, split, config.RANDOM_SEED
    )

    if len(train_ds) == 0:
        print("Error: train split is empty.")
        return

    # 2. Sampler & Dataloaders
    sampler = None
    if getattr(config, "TRAIN_USE_WEIGHTED_SAMPLER", False):
        labels = helper.labels_for_subset(train_ds)
        n_pos = sum(1 for y in labels if y >= 0.5)
        n_neg = len(labels) - n_pos
        if n_pos > 0 and n_neg > 0:
            w_pos, w_neg = 1.0 / n_pos, 1.0 / n_neg
            weights = [w_pos if y >= 0.5 else w_neg for y in labels]
            sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)

    num_workers = getattr(config, "TRAIN_NUM_WORKERS", 0)
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, sampler=sampler, 
        shuffle=(sampler is None), drop_last=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=num_workers)

    # 3. Model, Loss, Optimizer
    model = BioLattice3DResNet().to(device)
    pos_weight = None
    if getattr(config, "TRAIN_USE_CLASS_POS_WEIGHT", False):
        labels = helper.labels_for_subset(train_ds)
        n_pos = sum(1 for y in labels if y >= 0.5)
        n_neg = len(labels) - n_pos
        if n_pos > 0 and n_neg > 0:
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

    criterion = FocalLoss(alpha=config.FOCAL_LOSS_ALPHA, gamma=config.FOCAL_LOSS_GAMMA, pos_weight=pos_weight).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.ADAMW_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.ONECYCLE_MAX_LR, steps_per_epoch=len(train_loader), epochs=config.EPOCHS)

    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)
    train_labels = helper.labels_for_subset(train_ds)
    val_labels = helper.labels_for_subset(val_ds)
    n_tr_pos = sum(1 for y in train_labels if y >= 0.5)
    n_tr_neg = len(train_labels) - n_tr_pos
    n_va_pos = sum(1 for y in val_labels if y >= 0.5)
    n_va_neg = len(val_labels) - n_va_pos
    class_balance_log = (
        f"train_label_0: {n_tr_neg} | train_label_1: {n_tr_pos}",
        f"val_label_0: {n_va_neg} | val_label_1: {n_va_pos}",
    )

    # 4. Training Cycle
    run_log = RunLogWriter.create_new(kind="training") if getattr(config, "TRAIN_LOG_WRITE_TXT", True) else None
    trainer = BioLatticeTrainer(model, train_loader, val_loader, criterion, optimizer, scheduler, device, run_log)

    t_loop_mono = None
    epochs_completed = 0
    early_stopped = False
    loop_error = None

    if run_log:
        start_iso = datetime.now().isoformat(timespec="seconds")
        run_log.write_training_header(
            start_iso=start_iso,
            device_str=str(device),
            n_dataset_train=len(dataset_train),
            n_dataset_val=len(dataset_val),
            excluded_ids_audit=excluded_ids,
            train_size=train_size,
            val_size=val_size,
            train_batch_size=config.BATCH_SIZE,
            train_drop_last=True,
            train_batches_per_epoch=num_train_batches,
            val_batches_per_epoch=num_val_batches,
            class_balance_lines=class_balance_log,
        )
        print(f"Training run log: {run_log.path}", flush=True)
        
        setup_dur = time.monotonic() - t_start_setup
        run_log.write_setup_stats(setup_dur)
        print(f"Setup completed in {setup_dur:.2f}s", flush=True)

    try:
        t_loop_mono = time.monotonic()
        for epoch in range(config.EPOCHS):
            t_epoch_start = time.monotonic()
            
            t_loss = trainer.train_epoch()
            v_loss = trainer.validate()
            
            t_epoch_dur = time.monotonic() - t_epoch_start
            lr = scheduler.get_last_lr()[0]

            improved = (not math.isnan(v_loss)) and (v_loss < (trainer.best_val_loss - config.EARLY_STOPPING_MIN_DELTA))
            if improved:
                trainer.best_val_loss = v_loss
                trainer.epochs_without_improve = 0
                trainer.save_checkpoint(config.PATH_MODEL_WEIGHTS)
            elif not math.isnan(v_loss):
                trainer.epochs_without_improve += 1

            epochs_completed = epoch + 1

            status_line = (
                f"Epoch [{epoch+1}/{config.EPOCHS}] | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | "
                f"LR: {lr:.6f} | Improved: {improved} | {t_epoch_dur:.1f}s"
            )
            print(status_line, flush=True)
            if run_log:
                run_log.append_epoch_line(
                    epoch=epochs_completed,
                    epochs_max=config.EPOCHS,
                    train_loss=t_loss,
                    val_loss=v_loss,
                    lr=float(lr),
                    saved_best=improved,
                    best_val_loss=trainer.best_val_loss,
                    epochs_without_improve=trainer.epochs_without_improve,
                    duration_sec=t_epoch_dur,
                )

            if (epoch + 1) >= config.EARLY_STOPPING_START_EPOCH and trainer.epochs_without_improve >= config.EARLY_STOPPING_PATIENCE:
                msg = (
                    f"Early stopping triggered at epoch {epoch+1}: "
                    f"no validation improvement > {config.EARLY_STOPPING_MIN_DELTA:.4f} for "
                    f"{config.EARLY_STOPPING_PATIENCE} consecutive epochs."
                )
                print(msg, flush=True)
                if run_log:
                    run_log.append_note(msg)
                early_stopped = True
                break
        print("Model training completed.", flush=True)
    except Exception as e:
        loop_error = str(e)
        print(f"Training failed: {e}", flush=True)
        trainer.log_to_file(f"note: training exception: {loop_error}")
        raise
    finally:
        if trainer and trainer.run_log and t_loop_mono is not None:
            end_iso = datetime.now().isoformat(timespec="seconds")
            duration = time.monotonic() - t_loop_mono
            status = "failed" if loop_error else ("early_stop" if early_stopped else "completed")
            trainer.run_log.write_training_footer(
                end_iso=end_iso,
                duration_sec=duration,
                epochs_completed=epochs_completed,
                best_val_loss=trainer.best_val_loss,
                early_stopped=early_stopped,
                status=status,
                extra=loop_error or "",
            )

if __name__ == "__main__":
    train_model()
