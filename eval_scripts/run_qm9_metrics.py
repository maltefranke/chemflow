#!/usr/bin/env python3
"""Script to compute metrics on the entire QM9 dataset."""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from torchmetrics import MetricCollection

from chemflow.dataset.qm9 import QM9Charges
from chemflow.metrics import (
    init_metrics,
    calc_atom_stabilities,
    Validity,
)
from chemflow.utils import rdkit_utils as chemflowRD
from chemflow.utils import z_to_atom_types


def main():
    parser = argparse.ArgumentParser(description="Compute metrics on QM9 dataset")
    parser.add_argument(
        "--root",
        type=str,
        default="/cluster/project/krause/frankem/chemflow/data/qm9",
        help="Root directory for QM9 dataset",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing molecules (default: 1000)",
    )
    parser.add_argument(
        "--validity-only",
        action="store_true",
        help="Only compute validity metrics (faster)",
    )
    args = parser.parse_args()

    # Load QM9 dataset
    print(f"Loading QM9 dataset from {args.root}...")
    dataset = QM9Charges(root=args.root)
    print(f"Loaded {len(dataset)} molecules")

    # Initialize metrics
    print("Initializing metrics...")
    if args.validity_only:
        # Only compute validity metrics
        metrics = MetricCollection(
            {
                "validity": Validity(),
                "connected-validity": Validity(connected=True),
            },
            compute_groups=False,
        )
        stability_metrics = None
        print("Computing validity metrics only")
    else:
        metrics, stability_metrics, _, _ = init_metrics()

    # Process molecules in batches
    print(f"Processing molecules in batches of {args.batch_size}...")

    batch_mols = []
    batch_stabs = []

    for i in tqdm(range(len(dataset)), desc="Converting to RDKit molecules"):
        data = dataset[i]

        # Extract 3D structure data
        coords = data.pos.numpy()  # Shape: [n_atoms, 3]
        atomic_numbers = data.z.numpy()  # Shape: [n_atoms]
        edge_index = data.edge_index.numpy()  # Shape: [2, n_edges]
        edge_attr = data.edge_attr.numpy()  # Shape: [n_edges, 4]
        # Check if charges exist
        charges = (
            data.charges.numpy() if hasattr(data, "charges") else None
        )  # Shape: [n_atoms]

        # Convert atomic numbers to atom symbols
        atom_symbols = z_to_atom_types(atomic_numbers.tolist())

        # Convert edge_attr from one-hot to bond type indices
        # edge_attr: 0=single, 1=double, 2=triple, 3=aromatic
        # IDX_BOND_MAP expects: 1=single, 2=double, 3=triple, 4=aromatic
        bond_type_indices = edge_attr.argmax(axis=1) + 1  # Convert 0-3 to 1-4

        # QM9 has undirected edges (each bond appears twice: (i,j) and (j,i))
        # We need to deduplicate by keeping only one direction (start < end)
        start_indices = edge_index[0]
        end_indices = edge_index[1]

        # Create bonds array: [start, end, bond_type]
        bonds = np.column_stack(
            [
                start_indices,  # start indices
                end_indices,  # end indices
                bond_type_indices,  # bond types (1-4)
            ]
        )

        # Convert charges if available
        charges_array = charges if charges is not None else None

        # Create RDKit molecule from 3D structure
        mol = chemflowRD.mol_from_atoms(
            coords=coords,
            tokens=atom_symbols,
            bonds=bonds,
            charges=charges_array,
            sanitise=True,
        )

        if mol is not None:
            batch_mols.append(mol)
            # Calculate atom stabilities only if not validity-only mode
            if not args.validity_only:
                stabilities = calc_atom_stabilities(mol)
                batch_stabs.append(stabilities)

        # Update metrics in batches
        if len(batch_mols) >= args.batch_size:
            metrics.update(batch_mols)
            if stability_metrics is not None:
                stability_metrics.update(batch_stabs)
            batch_mols = []
            if not args.validity_only:
                batch_stabs = []

    # Process remaining molecules
    if batch_mols:
        metrics.update(batch_mols)
        if stability_metrics is not None:
            stability_metrics.update(batch_stabs)

    # Compute final metrics
    print("\nComputing final metrics...")
    results = metrics.compute()
    if stability_metrics is not None:
        stab_results = stability_metrics.compute()
        all_results = {**results, **stab_results}
    else:
        all_results = results

    # Print results
    print("\n" + "=" * 60)
    print("METRIC RESULTS FOR QM9 DATASET")
    print("=" * 60)
    for key, value in sorted(all_results.items()):
        # Handle tensor values
        if hasattr(value, "item"):
            print(f"{key}: {value.item():.6f}")
        elif isinstance(value, (int, float)):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print("=" * 60)

    # Save results to file
    output_file = Path(args.root) / "metrics_results.txt"
    print(f"\nSaving results to {output_file}...")
    with open(output_file, "w") as f:
        f.write("METRIC RESULTS FOR QM9 DATASET\n")
        f.write("=" * 60 + "\n")
        for key, value in sorted(all_results.items()):
            # Handle tensor values
            if hasattr(value, "item"):
                f.write(f"{key}: {value.item():.6f}\n")
            elif isinstance(value, (int, float)):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("=" * 60 + "\n")

    print("Done!")


if __name__ == "__main__":
    main()
