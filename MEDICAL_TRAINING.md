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

If you hit a `CalledProcessError` while training:

* First, re-run with `--embed-extra -debug -ndproc 0` to print the underlying
  `embed.py` stack trace and disable multiprocessing.
* If the error mentions `BatchedDataset` or a missing Cython extension,
  install the compiled extensions with `python setup.py build_ext --inplace`
  and retry. The helper now falls back to a pure-Python loader for CSV edge
  lists, but the compiled version is faster on large datasets.
* If you see `ModuleNotFoundError: No module named "hype.adjacency_matrix_dataset"`
  when training on an HDF5 adjacency matrix, the Cython module was not built.
  Rebuild with `python setup.py build_ext --inplace` (or reinstall the package)
  so `AdjacencyDataset` becomes available.

The passthrough flag can forward any `embed.py` option (e.g., `-gpu -1` to
force CPU or `-fresh` to discard old checkpoints).

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

## 5) `medical_embeddings.py` code walkthrough

If you want to adapt or audit the helper script, here is how it works end to
end:

* **Normalization and parsing**: `_normalize_code` strips dots/whitespace and
  upper-cases ICD/ATC tokens, while `_load_records` streams the JSONL input and
  safely pulls the diagnosis/procedure/medication arrays using configurable
  keys (extra fields are ignored).
* **Hierarchy expansion**: `_prefixes_for_code` and `_determine_parent` expand
  each code into its hierarchical prefixes (ICD uses 3/4/5 characters; ATC
  uses 1/3/4/5/7) and pick the closest observed prefix as a parent. When no
  prefix exists, the code attaches to a synthetic root label.
* **Edge construction**: `_build_hierarchy_edges` counts how often each code
  appears, seeds the node set with all valid prefixes, and emits `parent,child`
  edges weighted by the observed frequency (minimum 1). `_write_edge_csv`
  writes these edges as `id1,id2,weight` CSV files in the specified output
  directory.
* **Command-line entry point**: `build_edge_lists` orchestrates the above steps
  for diagnoses, procedures, and medications, returning the paths to each CSV.
  When `--train` is passed, `_run_training` calls `embed.py` separately for each
  modality using the provided hyperparameters and writes checkpoints to
  `--checkpoint-dir`.
* **CLI options**: The script accepts custom JSON keys, output/checkpoint
  locations, embedding dimension, manifold choice, learning rate, epochs,
  batch size, negative samples, burn-in, and dampening. See `python
  medical_embeddings.py --help` for defaults and descriptions.

These pieces together turn a structured JSONL file into reproducible Poincaré
training datasets and (optionally) trained embeddings for the three clinical
code systems.
