# TPU LM Experiments

Use these scripts after ABCDigits has confirmed the architecture. These runs train on real LM text data.

## Data

`train_lm.py` accepts:

- `.txt` files
- `.jsonl` files with a text field
- `.npy` files containing a 1D token id array
- Hugging Face dataset repos via `--hf-repo`

Use `--token-cache` to tokenize once and reuse the cached `.npy` across sweeps. Text records are packed by concatenating tokenized records with EOS separators before fixed-length chunks are sampled, so short records still fill `--sequence-length`.

Local JSONL:

```bash
--data-path /kaggle/input/YOUR_DATA/train.jsonl --text-field text
```

Hugging Face dataset:

```bash
--hf-repo wikimedia/wikipedia --hf-config 20231101.th --hf-split train --text-field text
```

## Stage 1: Choose Model Size

This keeps `S=128`, `T=4` and tries model sizes around 1B parameters.

```bash
uv run python experiments-tpu/run_stage.py \
  --stage model-size \
  --hf-repo wikimedia/wikipedia \
  --hf-config 20231101.th \
  --hf-split train \
  --text-field text \
  --token-cache /kaggle/working/tachia_tokens.npy \
  --sequence-length 512 \
  --steps 5000 \
  --batch-size 1 \
  --gradient-accumulation-steps 32
```

Default model-size candidates, estimated with the local vocab size and tied embeddings:

- `embed_dim=1408`, `layers=24`, `heads=16`: about 0.88B params
- `embed_dim=1536`, `layers=22`, `heads=16`: about 0.97B params
- `embed_dim=1536`, `layers=24`, `heads=16`: about 1.04B params

Pick the largest one with acceptable memory and speed.

## Stage 2: Sweep Slots

After choosing model size, keep `T=4` and sweep `S`.

```bash
uv run python experiments-tpu/run_stage.py \
  --stage slots \
  --hf-repo wikimedia/wikipedia \
  --hf-config 20231101.th \
  --hf-split train \
  --text-field text \
  --token-cache /kaggle/working/tachia_tokens.npy \
  --embed-dim 1536 \
  --layers 24 \
  --heads 16 \
  --model-name e1536-l24-h16 \
  --slots 64 128 256 512 \
  --sequence-length 512 \
  --steps 5000 \
  --batch-size 1 \
  --gradient-accumulation-steps 32
```

Choose `S` using validation loss, training speed, and selector metrics.

## Stage 3: Sweep Temperature

After choosing model size and `S`, keep them fixed and sweep `T`.

```bash
uv run python experiments-tpu/run_stage.py \
  --stage temperature \
  --hf-repo wikimedia/wikipedia \
  --hf-config 20231101.th \
  --hf-split train \
  --text-field text \
  --token-cache /kaggle/working/tachia_tokens.npy \
  --embed-dim 1536 \
  --layers 24 \
  --heads 16 \
  --model-name e1536-l24-h16 \
  --base-slots 256 \
  --temperatures 2 4 8 16 \
  --sequence-length 512 \
  --steps 5000 \
  --batch-size 1 \
  --gradient-accumulation-steps 32
```

Avoid `T=32` until lower temperatures are clearly too diffuse.

## Memory Defaults

The scripts default to GaLore AdamW and rematerialization:

```bash
--optimizer galore_adamw --galore-rank 128 --galore-update-proj-gap 200 --remat
```

GaLore reduces optimizer-state memory. It does not reduce selector activation memory, so keep per-device `--batch-size` small and use `--gradient-accumulation-steps` for effective batch size.
