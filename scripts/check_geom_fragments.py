"""Quick check for fragmented molecules in a processed GEOM LMDB split."""

import argparse
import os

from torch_geometric.utils import to_networkx
import networkx as nx
from tqdm import tqdm

from chemflow.dataset.geom import LEN_KEY, mol_from_bytes, open_read_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=os.path.join(os.environ.get("PROJECT_ROOT", "."), "data", "geom"),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--show", type=int, default=10, help="Examples to print")
    args = parser.parse_args()

    lmdb_path = os.path.join(args.root, "processed", f"{args.split}_data.lmdb")
    env = open_read_env(lmdb_path)

    total = 0
    n_frag = 0
    examples = []
    try:
        with env.begin() as txn, txn.cursor() as cur:
            for key, value in tqdm(cur, desc=f"Scanning {args.split}"):
                if bytes(key) == LEN_KEY:
                    continue
                total += 1
                data = mol_from_bytes(bytes(value))
                g = to_networkx(data, to_undirected=True)
                n_components = nx.number_connected_components(g)
                if n_components > 1:
                    n_frag += 1
                    if len(examples) < args.show:
                        examples.append((total - 1, n_components, data.smiles))
    finally:
        env.close()

    print(f"\nTotal molecules : {total}")
    print(f"Fragmented      : {n_frag} ({100 * n_frag / max(total, 1):.4f}%)")
    if examples:
        print("\nExamples:")
        for idx, n_components, smi in examples:
            print(f"  [{idx}] components={n_components}  {smi}")


if __name__ == "__main__":
    main()
