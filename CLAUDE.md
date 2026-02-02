# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MolGPT is a molecular generation framework using a GPT (Transformer-Decoder) model for SMILES generation. It supports:
- Unconditional and conditional molecular generation
- Property-conditional generation (QED, LogP, SAS)
- Scaffold-based conditional generation
- Evidential Deep Learning (EDL) for uncertainty quantification

## Common Commands

### Training

```bash
# Activate environment
conda activate molgpt_dos

# EDL training (primary method, multi-GPU)
cd train/
./train_edl.sh

# Custom training
./train_edl.sh --run_name my_exp --max_epochs 50 --batch_size 256 --num_gpus 4 --gpu_ids "0,1,2,3"

# Single GPU training
./train_edl.sh --num_gpus 1 --gpu_ids "0"

# Standard training (no EDL)
./train_edl.sh --no_edl
```

### Testing

```bash
# EDL unit tests
python train/test_edl.py

# Training tests
python train/test_training.py
```

### Generation

```bash
python generate/generate.py \
    --model_weight ../cond_gpt/weights/model_name.pt \
    --data_name moses2 \
    --csv_name output_name \
    --gen_size 10000
```

### Monitoring

```bash
tail -f train/logs/edl_molgpt_*.log   # Training logs
watch -n 1 nvidia-smi                  # GPU usage
kill $(cat train/train.pid)            # Stop training
```

## Architecture

### Core Components

- `train/model.py` - GPT architecture with optional EDL output head
- `train/trainer.py` - Training loop with Hugging Face Accelerate for multi-GPU
- `train/edl_loss.py` - EDL loss functions (MSE and Log variants)
- `train/uncertainty.py` - Uncertainty metrics (epistemic, aleatoric, total)
- `train/dataset.py` - SmileDataset class for SMILES tokenization
- `generate/generate.py` - Inference and molecule generation

### Model Configuration

Default architecture: 8 layers, 8 heads, 256 embedding dim
- MOSES: vocab_size=26, block_size=72
- GuacaMol: vocab_size=94, block_size=100

### EDL Integration

The model outputs evidence via softplus activation, parameterizing a Dirichlet distribution:
- α = evidence + 1
- Predicted probability: p = α / S (where S = sum of α)
- Uncertainty: u = K / S (where K = num classes)

Loss = MSE/Log loss + KL divergence (with annealing over first N epochs)

### Multi-GPU Training

Uses Hugging Face Accelerate with data parallelism. Effective batch size = batch_size × num_gpus.

### Key Directories

- `train/` - Training scripts and core modules
- `train/logs/` - Training log files
- `cond_gpt/weights/` - Saved model checkpoints
- `datasets/` - MOSES and GuacaMol datasets (CSV format)
- `generate/` - Generation scripts
- `evaluate/` - Evaluation metrics (GuacaMol, MOSES benchmarks)

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_epochs` | 20 | Training epochs |
| `--batch_size` | 512 | Per-GPU batch size |
| `--learning_rate` | 6e-4 | Initial learning rate |
| `--n_layer` | 8 | Transformer layers |
| `--n_embd` | 256 | Embedding dimension |
| `--edl_loss_type` | mse | EDL loss type (mse or log) |
| `--edl_annealing_step` | 10 | KL annealing epochs |

## Citation

Bagal, Viraj; Aggarwal, Rishal; Vinod, P. K.; Priyakumar, U. Deva (2021): MolGPT: Molecular Generation using a Transformer-Decoder Model. ChemRxiv. https://doi.org/10.26434/chemrxiv.14561901.v1
