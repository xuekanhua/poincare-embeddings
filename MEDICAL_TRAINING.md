# Training hyperbolic embeddings for ICD-9/ICD-9-CM and ATC codes

This guide explains how to transform clinical code data into the edge-list
format expected by `embed.py` and how to train three independent Poincaré
embedding models for diagnoses, procedures, and medications.

## 1) Prepare the data file

Supply a newline-delimited JSON (`.jsonl`) file where each line represents one
visit or patient and includes arrays of codes:

```json
{"diagnoses": ["9992", "4239", "5119", "V1259"], "procedures": ["51.19"], "medications": ["B05C", "M01A", "A02A", "J01M"]}
```

* Dots in ICD-9/ICD-9-CM codes are optional; they are removed automatically.
* ICD codes are grouped by prefixes of length 3, 4, and 5 to reflect their
  hierarchy. ATC codes use the standard 1/3/4/5/7 character levels.

## 2) Build edge lists

Run the helper to expand code prefixes into a parent–child edge list per
modality:

```bash
python medical_embeddings.py --input data/records.jsonl --output-dir data/medical_edges
```

This produces CSV files such as `diagnoses_edges.csv`, `procedures_edges.csv`,
and `medications_edges.csv` with the columns `id1,id2,weight`, where `weight`
counts how often the child code appeared.

### Optional: run edge creation and training together

Add `--train` to launch the standard training loop right after edge generation:

```bash
python medical_embeddings.py \
  --input data/records.jsonl \
  --output-dir data/medical_edges \
  --checkpoint-dir checkpoints \
  --dim 50 --manifold lorentz --lr 1000 --epochs 50 --batchsize 4096 --negs 50 \
  --burnin 10 --dampening 0.75 --train
```

## 3) Train each embedding model manually (if you skipped `--train`)

Use `embed.py` with the edge list for each modality. Example (diagnoses):

```bash
python embed.py \
  -dset data/medical_edges/diagnoses_edges.csv \
  -checkpoint checkpoints/diagnoses.pth \
  -dim 50 -manifold lorentz -lr 1000 -epochs 50 -batchsize 4096 \
  -negs 50 -burnin 10 -dampening 0.75
```

Repeat with `procedures_edges.csv` and `medications_edges.csv` to obtain three
separate checkpoints. Adjust dimensions, learning rate, and other hyper-
parameters as needed for your dataset size and hardware.

## 4) Interpreting outputs

Each checkpoint (`*.pth`) stores the trained embeddings and metadata. The rows
in the `objects` list inside the checkpoint match the order of codes from the
edge list, enabling downstream nearest-neighbor or reconstruction evaluation via
existing tooling in this repository.
