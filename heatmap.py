import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from chess import RANK_NAMES, FILE_NAMES
import os

CHECKPOINT_PATH = "checkpoints/epoch_058.pt"
PIECE_NAMES = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"]
matplotlib.use("TkAgg")


def plot_psqt():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"File not found: {CHECKPOINT_PATH}")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = checkpoint["model"]

    mg_tables = state_dict["psqt_mg"][:6].round().numpy()
    eg_tables = state_dict["psqt_eg"][:6].round().numpy()

    for i in range(6):
        fig, (ax_mg, ax_eg) = plt.subplots(1, 2, figsize=(14, 6))
        fig.canvas.manager.set_window_title(f"Análisis PSQT: {PIECE_NAMES[i]}")

        mg_data = mg_tables[i].reshape(8, 8)
        eg_data = eg_tables[i].reshape(8, 8)

        sns.heatmap(
            mg_data,
            ax=ax_mg,
            annot=True,
            fmt=".0f",
            cmap="RdYlGn",
            center=mg_data.mean(),
            xticklabels=FILE_NAMES,
            yticklabels=RANK_NAMES,
            cbar_kws={"label": "Centipawns"},
        )
        ax_mg.set_title(f"{PIECE_NAMES[i]} - Middle Game")
        ax_mg.invert_yaxis()

        sns.heatmap(
            eg_data,
            ax=ax_eg,
            annot=True,
            fmt=".0f",
            cmap="RdYlGn",
            center=eg_data.mean(),
            xticklabels=FILE_NAMES,
            yticklabels=RANK_NAMES,
            cbar_kws={"label": "Centipawns"},
        )
        ax_eg.set_title(f"{PIECE_NAMES[i]} - End Game")
        ax_eg.invert_yaxis()

        plt.suptitle(
            f"Piece: {PIECE_NAMES[i]} (Checkpoint: {os.path.basename(CHECKPOINT_PATH)})",
            fontsize=14,
        )

    plt.show()


if __name__ == "__main__":
    plot_psqt()
