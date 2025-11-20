"""Utilities for building Poincaré training data for clinical code hierarchies.

This script prepares edge lists for ICD-9/ICD-9-CM and ATC codes using the
prefix structure described in "Poincaré Embeddings for Learning Hierarchical
Representations" (Nickel & Kiela, 2017). It can also launch `embed.py` to train
separate embeddings for diagnoses, procedures, and medications once the edge
lists are written to disk.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

ICD9_LEVEL_LENGTHS: Tuple[int, ...] = (3, 4, 5)
ATC_LEVEL_LENGTHS: Tuple[int, ...] = (1, 3, 4, 5, 7)
ROOT_PREFIX = "__ROOT__"


def _normalize_code(code: str) -> str:
    """Normalize incoming codes by stripping dots/whitespace and upper-casing."""
    return code.replace(".", "").strip().upper()


def _load_records(
    path: Path, diag_key: str, procedure_key: str, medication_key: str
) -> Iterator[Dict[str, Sequence[str]]]:
    """Yield dictionaries from a JSON-lines file.

    Each line is expected to contain arrays of diagnosis, procedure, and
    medication codes under configurable keys. Extra keys are ignored.
    """

    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc

            yield {
                diag_key: record.get(diag_key, []) or [],
                procedure_key: record.get(procedure_key, []) or [],
                medication_key: record.get(medication_key, []) or [],
            }


def _prefixes_for_code(code: str, level_lengths: Tuple[int, ...]) -> List[str]:
    return [code[:length] for length in level_lengths if len(code) >= length]


def _determine_parent(
    code: str, level_lengths: Tuple[int, ...], nodes: Sequence[str]
) -> str | None:
    """Pick the nearest valid prefix as the parent for `code` if present."""

    candidate_lengths = [length for length in level_lengths if length < len(code)]
    for length in sorted(candidate_lengths, reverse=True):
        candidate = code[:length]
        if candidate in nodes:
            return candidate
    return None


def _build_hierarchy_edges(
    codes: Iterable[str],
    level_lengths: Tuple[int, ...],
    root_label: str,
) -> List[Tuple[str, str, float]]:
    """Build parent-child edges from a collection of codes.

    The algorithm relies solely on prefix overlap to approximate the hierarchy.
    Each edge is weighted by the number of times the child code appeared in the
    dataset (minimum weight of 1). Codes with no valid prefix fall back to the
    provided root label.
    """

    counts = Counter(codes)
    # Seed the node set with all prefixes that match the configured levels.
    node_set = set()
    for code in counts:
        node_set.update(_prefixes_for_code(code, level_lengths))
    # Ensure level nodes exist even if not observed directly in the data.
    for code in list(node_set):
        counts.setdefault(code, 0)

    edges: List[Tuple[str, str, float]] = []
    for code, weight in counts.items():
        parent = _determine_parent(code, level_lengths, node_set)
        parent = parent if parent is not None else root_label
        edges.append((parent, code, float(weight or 1)))
    return edges


def _write_edge_csv(path: Path, edges: List[Tuple[str, str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("id1,id2,weight\n")
        for parent, child, weight in edges:
            handle.write(f"{parent},{child},{weight}\n")


def _collect_codes(records: Iterable[Dict[str, Sequence[str]]], key: str) -> List[str]:
    codes: List[str] = []
    for record in records:
        codes.extend(_normalize_code(code) for code in record.get(key, []))
    return codes


def build_edge_lists(
    input_path: Path,
    output_dir: Path,
    diag_key: str = "diagnoses",
    procedure_key: str = "procedures",
    medication_key: str = "medications",
) -> Dict[str, Path]:
    """Generate CSV edge lists for diagnoses, procedures, and medications."""

    records = list(_load_records(input_path, diag_key, procedure_key, medication_key))
    if not records:
        raise ValueError(f"No records found in {input_path}")

    outputs: Dict[str, Path] = {}
    for label, key, levels in (
        ("diagnoses", diag_key, ICD9_LEVEL_LENGTHS),
        ("procedures", procedure_key, ICD9_LEVEL_LENGTHS),
        ("medications", medication_key, ATC_LEVEL_LENGTHS),
    ):
        codes = _collect_codes(records, key)
        if not codes:
            logging.warning("No codes found for %s; skipping edge list.", label)
            continue
        edges = _build_hierarchy_edges(codes, levels, f"{ROOT_PREFIX}_{label}")
        output_path = output_dir / f"{label}_edges.csv"
        _write_edge_csv(output_path, edges)
        outputs[label] = output_path
        logging.info("Wrote %s edges to %s", label, output_path)
    return outputs


def _run_training(
    dset_path: Path,
    checkpoint: Path,
    dim: int,
    manifold: str,
    lr: float,
    epochs: int,
    batchsize: int,
    negs: int,
    burnin: int,
    dampening: float,
    extra_args: Sequence[str] | None = None,
) -> None:
    """Invoke the existing embed.py CLI for a single dataset."""

    cmd = [
        "python",
        str(Path(__file__).parent / "embed.py"),
        "-dset",
        str(dset_path),
        "-checkpoint",
        str(checkpoint),
        "-dim",
        str(dim),
        "-manifold",
        manifold,
        "-lr",
        str(lr),
        "-epochs",
        str(epochs),
        "-batchsize",
        str(batchsize),
        "-negs",
        str(negs),
        "-burnin",
        str(burnin),
        "-dampening",
        str(dampening),
    ]
    if extra_args:
        cmd.extend(extra_args)
    logging.info("Launching training: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - runtime aid
        logging.error("embed.py failed with return code %s", exc.returncode)
        if exc.stdout:
            logging.error("stdout:\n%s", exc.stdout)
        if exc.stderr:
            logging.error("stderr:\n%s", exc.stderr)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build edge lists for ICD/ATC codes and optionally train embeddings."
    )
    parser.add_argument("--input", required=True, type=Path, help="JSONL records file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/medical_edges"),
        help="Directory to store generated edge CSV files.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Where to store trained embeddings if --train is enabled.",
    )
    parser.add_argument("--diagnosis-key", default="diagnoses")
    parser.add_argument("--procedure-key", default="procedures")
    parser.add_argument("--medication-key", default="medications")
    parser.add_argument("--train", action="store_true", help="Run embed.py after building edges")
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--manifold", default="lorentz")
    parser.add_argument("--lr", type=float, default=1000.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batchsize", type=int, default=2048)
    parser.add_argument("--negs", type=int, default=50)
    parser.add_argument("--burnin", type=int, default=10)
    parser.add_argument("--dampening", type=float, default=0.75)
    parser.add_argument(
        "--embed-extra",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to embed.py (e.g., --embed-extra -ndproc 0 -fresh)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    edge_paths = build_edge_lists(
        args.input,
        args.output_dir,
        diag_key=args.diagnosis_key,
        procedure_key=args.procedure_key,
        medication_key=args.medication_key,
    )

    if args.train:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for label, dset_path in edge_paths.items():
            checkpoint = args.checkpoint_dir / f"{label}.pth"
            _run_training(
                dset_path,
                checkpoint,
                dim=args.dim,
                manifold=args.manifold,
                lr=args.lr,
                epochs=args.epochs,
                batchsize=args.batchsize,
                negs=args.negs,
                burnin=args.burnin,
                dampening=args.dampening,
                extra_args=args.embed_extra,
            )


if __name__ == "__main__":
    main()
