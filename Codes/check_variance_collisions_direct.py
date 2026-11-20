from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import json
import math
import multiprocessing as mp
from pathlib import Path
import time
import traceback

import networkx as nx

from check_tree_collision import (
    ParentPackedPolynomialEngine,
    iter_parent_codes,
)

_ENGINE = None
_FIXED_LENGTH = None


def exact_variance_key(coeffs, n, fixed_length=None):
    length = (len(coeffs) + n) if fixed_length is None else fixed_length
    s1 = sum(coeffs)
    s2 = sum(c * c for c in coeffs)
    num = length * s2 - s1 * s1
    den = length * length
    g = math.gcd(num, den)
    return (num // g, den // g)


def _init_worker(n, fixed_length):
    global _ENGINE, _FIXED_LENGTH
    _ENGINE = ParentPackedPolynomialEngine(n)
    _FIXED_LENGTH = fixed_length


def _process_batch(batch):
    out = []
    n = _ENGINE.n
    for tree_index, code in batch:
        packed = _ENGINE.compute_parent_code(code)
        coeffs = _ENGINE.coefficients(packed)
        num, den = exact_variance_key(coeffs, n, _FIXED_LENGTH)
        out.append((tree_index, num, den))
    return out


def batched_trees(n, batch_size, max_trees=None):
    batch = []
    for item in iter_parent_codes(n, max_trees=max_trees):
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_variance_check(
    n,
    workers=None,
    batch_size=1000,
    pad_to_max_degree=False,
    max_trees=None,
    output_dir="variance_results",
):
    if n < 1:
        raise ValueError("مقدار n باید بزرگ‌تر یا مساوی 1 باشد.")

    if workers is None:
        workers = mp.cpu_count()

    expected_trees = 1 if n == 1 else nx.number_of_nonisomorphic_trees(n)
    fixed_length = (n * n + 1) if pad_to_max_degree else None

    print(
        f"Checking exact variances for n = {n} "
        f"(expected trees: {expected_trees:,}, workers: {workers})..."
    )
    started = time.monotonic()

    groups = defaultdict(list)
    errors = []
    total = 0

    if n == 1:
        groups[(1, 4)].append(0)
        total = 1
    else:
        pool = mp.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(n, fixed_length),
        )
        try:
            for results in pool.imap_unordered(
                _process_batch,
                batched_trees(n, batch_size, max_trees=max_trees),
                chunksize=1,
            ):
                for tree_index, num, den in results:
                    groups[(num, den)].append(tree_index)
                    total += 1
            pool.close()
            pool.join()
        except Exception:
            pool.terminate()
            pool.join()
            errors.append(traceback.format_exc())

    elapsed = time.monotonic() - started
    collisions = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
    
    verified_complete = False
    if not errors:
        if max_trees is None:
            if total == expected_trees:
                verified_complete = True
            else:
                errors.append(
                    f"Incomplete scan: checked {total:,} trees, expected {expected_trees:,}."
                )
        else:
            errors.append(
                f"Partial run (max_trees={max_trees}): full tree space was not checked."
            )

    all_variances_unique = verified_complete and (len(collisions) == 0)

    output_data = {
        "n": n,
        "workers_used": workers,
        "pad_to_max_degree": pad_to_max_degree,
        "expected_trees": expected_trees,
        "total_trees_checked": total,
        "verified_complete": verified_complete,
        "unique_exact_variances": len(groups),
        "all_variances_unique": all_variances_unique,
        "collision_groups_count": len(collisions),
        "elapsed_seconds": elapsed,
        "collisions": [
            {
                "group_id": idx,
                "exact_variance": f"{num}/{den}",
                "float_approximation": float(Fraction(num, den)),
                "tree_indices": tree_indices,
            }
            for idx, ((num, den), tree_indices) in enumerate(collisions.items(), start=1)
        ],
        "errors": errors,
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    output_filename = out_path / f"variance_collisions_n{n}.json"

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Completed in {elapsed:.2f} s.")
    if errors:
        print("!!! ERROR / INCOMPLETE RUN !!!")
        for err in errors:
            print(err)
    elif collisions:
        print(
            f"COLLISIONS FOUND: {len(collisions):,} group(s) among "
            f"{total:,} / {expected_trees:,} trees."
        )
    else:
        print(
            f"VERIFIED UNIQUE: All {total:,} / {expected_trees:,} trees "
            f"have strictly unique exact variances."
        )
    print(f"Saved to: {output_filename}")


if __name__ == "__main__":
    mp.freeze_support()

    start_n = 4
    end_n = 19

    for n_val in range(start_n, end_n + 1):
        print("\n" + "=" * 60)
        print(f"Starting variance check for n = {n_val}")
        print("=" * 60)
        run_variance_check(
            n=n_val,
            workers=16,
            batch_size=1000,
            pad_to_max_degree=False,
            output_dir="variance_results",
        )