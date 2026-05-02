import os
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

# =========================
# Constants
# =========================

PADDING_VALUE = 6
NUM_SQUARES = 64

BASE_MATERIAL = torch.tensor(
    [100.0, 320.0, 330.0, 500.0, 900.0, 0.0, 0.0], dtype=torch.float32
)
PHASE_WEIGHTS = torch.tensor([0.0, 1.0, 1.0, 2.0, 4.0, 0.0, 0.0], dtype=torch.float32)
MAX_PHASE = 24

BATCH_SIZE = 128
MOMENTUM = 0.9

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_num_threads(os.cpu_count() or 1)
torch.set_num_interop_threads(1)


# =========================
# Model
# =========================

class PSQTEval(nn.Module):
    def __init__(self, base_material, phase_weights, num_squares, max_phase):
        super().__init__()
        self.num_squares = num_squares
        self.max_phase = max_phase
        self.phase_weights: torch.Tensor
        self.register_buffer("phase_weights", phase_weights)

        NOISE = 0.5

        init = base_material.unsqueeze(1).repeat(1, num_squares)
        self.psqt_mg = nn.Parameter(init.clone())
        self.psqt_eg = nn.Parameter(init.clone())

        with torch.no_grad():
            self.psqt_mg += torch.randn_like(self.psqt_mg) * NOISE
            self.psqt_eg += torch.randn_like(self.psqt_eg) * NOISE

        self.K = nn.Parameter(torch.tensor([0.0090], dtype=torch.float32))

    def forward(self, pieces, squares, colors):
        mask = (pieces != PADDING_VALUE).float()

        # Flip vertically squares for blacks
        squares = torch.where(colors == 0, squares, squares ^ 56)

        # Batch indexing
        # To access flat tables
        idx = pieces.long() * self.num_squares + squares.long()

        mg = self.psqt_mg.view(-1)[idx]
        eg = self.psqt_eg.view(-1)[idx]

        sign = 1.0 - 2.0 * colors.float()
        mg_val = (mg * sign * mask).sum(dim=1)
        eg_val = (eg * sign * mask).sum(dim=1)

        # Phase calculateion
        phase = (self.phase_weights[pieces.long()] * mask).sum(dim=1)
        phase = (phase / self.max_phase).clamp(0.0, 1.0)

        # Tapered eval
        evaluation = phase * mg_val + (1.0 - phase) * eg_val

        # Apply logistic function with our learned K
        # To convert CP score into WDL probability
        return torch.sigmoid(self.K * evaluation)


# =========================
# Dataset
# =========================


def load_full_dataset(root_path):
    """Load and concat all dataset files into RAM"""
    pieces_list = []
    squares_list = []
    colors_list = []
    results_list = []

    # Search all chunk files
    chunk_files = sorted(glob.glob(os.path.join(root_path, "proc*_chunk_*.pt")))

    print(f"📦 Loading {len(chunk_files)} files in RAM...")

    for f in tqdm(chunk_files):
        data = torch.load(f, map_location="cpu")
        pieces_list.append(data["pieces"])
        squares_list.append(data["squares"])
        colors_list.append(data["colors"])
        results_list.append(data["result"])

    # Load into a single tensor
    # Use .contiguous() to make sure that the access is lineal
    full_dataset = TensorDataset(
        torch.cat(pieces_list).contiguous(),
        torch.cat(squares_list).contiguous(),
        torch.cat(colors_list).contiguous(),
        torch.cat(results_list).contiguous(),
    )

    return full_dataset


# =========================
# Checkpointing
# =========================


def save_checkpoint(epoch, model, optimizer):
    path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def load_latest_checkpoint(model, optimizer):
    files = sorted(f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt"))
    if not files:
        return 0

    path = os.path.join(CHECKPOINT_DIR, files[-1])
    ckpt = torch.load(path, map_location=DEVICE)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    print(f"✔ Resumed from {path}")
    return ckpt["epoch"] + 1


# =========================
# Weight normalization
# =========================


def material_consistency_loss(model, base_material):
    """
    Penalize deviations from the base material
    model: torch.Module which contains the PSQTs
    base_material: torch.Tensor (7,) with the base material values
    """
    mg_means = model.psqt_mg.mean(dim=1)
    eg_means = model.psqt_eg.mean(dim=1)

    diff_mg = mg_means[:6] - base_material[:6]
    diff_eg = eg_means[:6] - base_material[:6]

    return (diff_mg**2).sum() + (diff_eg**2).sum()


def psqt_deviation_loss(psqts_mg, psqts_eg, max_deviation=400.0, temperature=5.0):
    """
    A smooth, differentiable penalty for values exceeding max_deviation.
    Replaces the hard ReLU/Square cliff with a Softplus ceiling.
    """
    loss = 0.0
    for psqts in [psqts_mg, psqts_eg]:
        mean = psqts.mean(dim=1, keepdim=True)
        delta = psqts - mean

        # We calculate how much we exceed the limit
        # Softplus(x) = log(1 + exp(x))
        # As delta.abs() approaches max_deviation, the penalty starts to grow.
        # 'temperature' controls how sharp the turn is.
        # Higher temperature = sharper, more like your original ReLU.
        excess = nn.functional.softplus(delta.abs() - max_deviation, beta=temperature)

        # We square the soft excess to ensure a strong penalty for deep violations
        loss += excess.pow(2).mean()

    return loss


# =========================
# Training loop
# =========================


def train():
    print("Loading dataset in memory...")
    train_ds = load_full_dataset("dataset_pad/train")
    val_ds = load_full_dataset("dataset_pad/val")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("Initializing model and optimizer...")
    model = PSQTEval(BASE_MATERIAL, PHASE_WEIGHTS, NUM_SQUARES, MAX_PHASE).to(DEVICE)
    optimizer = torch.optim.Adagrad(
        [
            {
                "params": [model.psqt_mg, model.psqt_eg],
                "lr": 1.5,
            },
            {"params": [model.K], "lr": 0.01},
        ],
    )
    criterion = nn.MSELoss()

    print("Loading last checkpoint...")
    start_epoch = load_latest_checkpoint(model, optimizer)

    patience = 3  # How many epochs to wait for improvement
    min_delta = 0.00001  # Minimum change to qualify as an improvement
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print("--- Starting train ---")
    epoch = start_epoch
    while True:
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", unit="batch", leave=False)
        for batch_idx, (pieces, squares, colors, results) in enumerate(pbar):
            pieces = pieces.to(DEVICE)
            squares = squares.to(DEVICE)
            colors = colors.to(DEVICE)
            results = results.to(DEVICE)

            optimizer.zero_grad()
            evals = model(pieces, squares, colors)
            loss = (
                criterion(evals, results)
                + 0.05 * material_consistency_loss(model, BASE_MATERIAL)
                + 0.05 * psqt_deviation_loss(model.psqt_mg, model.psqt_eg)
            )
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            total_loss += current_loss

            if batch_idx % 50 == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{current_loss:.6f}",
                        "K": f"{model.K.item():.4f}",
                    }
                )

        avg_train = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(
                val_loader,
                desc=f"Validating epoch {epoch:03d}",
                unit="batch",
                leave=False,
            )
            for pieces, squares, colors, results in pbar:
                pieces = pieces.to(DEVICE)
                squares = squares.to(DEVICE)
                colors = colors.to(DEVICE)
                results = results.to(DEVICE)

                evals = model(pieces, squares, colors)
                val_loss += criterion(evals, results).item()

        val_loss /= len(val_loader)

        with torch.no_grad():
            print(
                f"PSQT MG std {model.psqt_mg.std():.3f} | "
                f"EG std {model.psqt_eg.std():.3f}"
            )

        print(
            f"Epoch {epoch:03d} | train {avg_train:.6f} | val {val_loss:.6f} | K {model.K.item():.4f}"
        )

        save_checkpoint(epoch, model, optimizer)

        if val_loss < (best_val_loss - min_delta):
            print(
                f"✔ Validation loss improved from {best_val_loss:.6f} to {val_loss:.6f}. Saving."
            )
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Only save the checkpoint if it's the best one we've seen
            save_checkpoint(epoch, model, optimizer)
        else:
            epochs_without_improvement += 1
            print(
                f"✖ No improvement. Early stopping counter: {epochs_without_improvement}/{patience}"
            )

        if epochs_without_improvement >= patience:
            print(f"🛑 Early stopping triggered at epoch {epoch}. Stopping training.")
            break

        epoch += 1

    print("Training complete.")


# =========================
# Entry
# =========================

if __name__ == "__main__":
    train()
