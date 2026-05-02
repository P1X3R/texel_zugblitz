# PSQT Texel Tuning Pipeline

A full training pipeline for a **tapered PSQT (Piece-Square Table) evaluation function** for a chess engine using a Texel-style approach with PyTorch.

What started as a “quick 400 LOC script” evolved into a complete system for:

* dataset generation from EPD
* efficient packed representations
* padded tensor datasets
* constrained PSQT learning
* checkpointed training with validation and early stopping

---

## Overview

This project implements a **modernized Texel tuning pipeline**:

1. Parse EPD positions with game outcomes
2. Convert positions into compact array representations
3. Pack and pad data for efficient batching
4. Train a **tapered midgame/endgame PSQT model**
5. Learn at the same time:

   * piece-square values
   * sigmoid scaling factor (K) for WDL prediction

---

## Key Ideas

* **Tapered Evaluation**
  Separate midgame (MG) and endgame (EG) PSQTs blended by phase.

* **Learned Logistic Scaling**
  A trainable parameter `K` converts centipawn evaluations into win/draw/loss probabilities.

* **Material Anchoring**
  PSQTs are regularized to stay consistent with base material values (avoid values to explode in one direction and finding out a pawn at e4 is worth 7000 cp).

* **Smooth Constraints**
  A soft penalty prevents PSQT values from drifting too far from reasonable ranges (and look better on the heatmap).

* **Use base `e` for sigmoid function**  
  Inspired by Ethereal's tuning method, this training uses a sigmoid with base `e`, letting the constant `K` absorb the `1/400` term from the original formula.

  Peter Österlund's formula:

  ```
  sigmoid = 1 / (1 + 10^(-K * E / 400))
  ```

  Modified (Ethereal-style):

  ```
  sigmoid = 1 / (1 + e^(-K * E))
  ```

> [!NOTE]
> Here E is an evaluation, and K is a coefficient computed to minimize an error function. 

---

## Pipeline

### 1. Dataset Creation (EPD -> NPZ)

```bash
python process_epd.py
```

* Parses EPD positions (from Zurichess dataset)
* Extracts:

  * pieces
  * squares
  * colors
  * results (W/D/L -> float)
* Outputs:

  * `train.npz`
  * `val.npz`

---

### 2. Padding & Chunking (NPZ -> PT)

```bash
python pad_dataset.py
```

* Converts variable-length positions into fixed-size tensors
* Applies padding for batching
* Splits into chunked `.pt` files
* Produces:

  * `dataset_pad/train/*.pt`
  * `dataset_pad/val/*.pt`

---

### 3. Training

```bash
python train.py
```

* Loads full dataset into memory
* Trains PSQT model using:

  * MSE loss vs WDL targets
  * material consistency regularization
  * PSQT deviation penalty
* Uses:

  * Adagrad optimizer
  * checkpointing
  * early stopping

---

## Model

The evaluation function:

* Computes MG and EG scores from PSQTs
* Blends them using a **phase function**
* Converts evaluation -> probability via:

```
P(win) = sigma(K * E)
```

Where:

* `K` is learned during training
* `E` is the model's evaluation in Centipawns (CP)

---

## Architecture

### Inputs per position

* `pieces`: piece types
* `squares`: board indices
* `colors`: white/black

### Core steps

1. Index PSQT tables
2. Apply color symmetry (flip for black)
3. Compute MG / EG scores
4. Compute phase
5. Blend -> final evaluation
6. Apply sigmoid

---

## Constraints & Regularization

To stabilize training:

### Material Consistency Loss

Keeps average PSQT values aligned with base material.

### PSQT Deviation Loss

Soft constraint preventing extreme square values using a smooth penalty.

---

## Training Features

* Batch training (PyTorch)
* Checkpointing per epoch
* Resume support
* Validation tracking
* Early stopping

---

## Why this exists

Traditional Texel tuning is often:

* minimally structured
* hard to scale
* loosely constrained

This project explores a more **systematic and extensible approach**:

* explicit data pipeline
* differentiable constraints
* learnable evaluation scaling

---

## Project Structure

```
.
├── process_epd.py        # EPD -> NPZ dataset
├── pad_dataset.py        # NPZ -> padded PT chunks
├── train.py              # Training loop
├── dataset_pad/          # Processed dataset
├── checkpoints/          # Saved models
```

---

## Notes

* Entire dataset is loaded into RAM during training
* Designed for experimentation, not production deployment even though Zugblitz's uses it.
* Assumes reasonably clean EPD input (look at Zurichess's format)
* Dataset padding could be done directly on EPD processing

---

## Origin Story

This was supposed to be a simple vibe coded Texel tuning script.

It wasn’t.

The code written by the was painfully slow and most of the time didn't even work. Therefore, I had to get my hands dirty with my little knowledge of Python and even less about PyTorch and Numpy I made this succeed. If you noticed the dataset padding could be done directly on EPD, this esign choice was taken precisely due to incorrect use of AI to "speed-up the development process" although it works effectively and remains stable. 

Tuning speed is surprising, converging on a local minimum within roughly 2 hours on an Intel Pentium Silver N5030 in comparison to the 6 hours it takes to CPW's tuning method on a 16-core Dell T620 computer. A completely expected effect but not at that scale.

This side-project is more an experiment on machine learning and correct AI use to actually speed-up the development process, learning that it's not an engineering and critical thinking replacement, but a reasoning copilot.
