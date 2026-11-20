import gc
import json
import multiprocessing as mp
from pathlib import Path
import queue
import time
import traceback

import networkx as nx

from tree_collision_search import PackedPolynomialEngine, graph6_text


def is_unimodal(coeffs):
    if not coeffs:
        return True
    i = 0
    length = len(coeffs)
    while i + 1 < length and coeffs[i] <= coeffs[i + 1]:
        i += 1
    while i + 1 < length and coeffs[i] >= coeffs[i + 1]:
        i += 1
    return i == length - 1


def worker_process(n, shard_id, num_shards, result_queue, stop_event):
    checked_count = 0
    try:
        engine = PackedPolynomialEngine(n)

        for i, T in enumerate(nx.nonisomorphic_trees(n)):
            if stop_event.is_set():
                break

            if i > 0 and i % 5000 == 0:
                gc.collect()

            if i % num_shards != shard_id:
                continue

            packed = engine.compute(T)
            coeffs = engine.coefficients(packed)
            checked_count += 1

            if not is_unimodal(coeffs):
                result_queue.put(("FOUND", shard_id, i, graph6_text(T), list(coeffs)))
                stop_event.set()
                return

        result_queue.put(("DONE", shard_id, checked_count))

    except Exception:
        result_queue.put(("ERROR", shard_id, traceback.format_exc()))
        stop_event.set()


def find_first_non_unimodal(n, workers=None, output_dir="unimodal_results"):
    if n < 1:
        raise ValueError("The value of n is invalid.")

    if workers is None:
        workers = mp.cpu_count()

    expected_trees = 1 if n == 1 else nx.number_of_nonisomorphic_trees(n)
    print(f"Searching for n = {n} (expected trees: {expected_trees:,}, workers: {workers})...")
    start_time = time.time()

    if n == 1:
        elapsed = time.time() - start_time
        output_data = {
            "n": 1,
            "workers_used": 1,
            "expected_trees": 1,
            "total_trees_checked": 1,
            "verified_complete": True,
            "elapsed_seconds": elapsed,
            "found": False,
            "tree_index": None,
            "graph6": None,
            "coefficients": None,
            "errors": [],
        }
    else:
        stop_event = mp.Event()
        result_queue = mp.Queue()

        pool = []
        for shard_id in range(workers):
            p = mp.Process(
                target=worker_process,
                args=(n, shard_id, workers, result_queue, stop_event),
            )
            p.start()
            pool.append(p)

        finished_shards = set()
        worker_errors = []
        found_result = None
        total_checked = 0

        while len(finished_shards) < workers and found_result is None and not worker_errors:
            try:
                msg = result_queue.get(timeout=0.2)
                msg_type = msg[0]

                if msg_type == "FOUND":
                    _, shard_id, index, g6, coeffs = msg
                    if is_unimodal(coeffs):
                        raise AssertionError(f"False positive received from shard {shard_id} at index {index}")
                    found_result = (index, g6, coeffs)
                    stop_event.set()
                    break

                elif msg_type == "DONE":
                    _, shard_id, count = msg
                    finished_shards.add(shard_id)
                    total_checked += count

                elif msg_type == "ERROR":
                    _, shard_id, err_text = msg
                    worker_errors.append(f"Shard {shard_id} error:\n{err_text}")
                    stop_event.set()
                    break

            except queue.Empty:
                for shard_id, p in enumerate(pool):
                    if (
                        not p.is_alive()
                        and p.exitcode not in (0, None)
                        and shard_id not in finished_shards
                    ):
                        worker_errors.append(
                            f"Shard {shard_id} terminated unexpectedly with exitcode={p.exitcode}"
                        )
                        stop_event.set()
                        break

        for p in pool:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
                p.join()

        elapsed = time.time() - start_time
        verified_complete = False
        if found_result is None and not worker_errors:
            if total_checked == expected_trees:
                verified_complete = True
            else:
                worker_errors.append(
                    f"Incomplete scan: checked {total_checked:,} trees, expected {expected_trees:,}."
                )

        output_data = {
            "n": n,
            "workers_used": workers,
            "expected_trees": expected_trees,
            "total_trees_checked": total_checked,
            "verified_complete": verified_complete,
            "elapsed_seconds": elapsed,
            "found": found_result is not None,
            "tree_index": found_result[0] if found_result else None,
            "graph6": found_result[1] if found_result else None,
            "coefficients": found_result[2] if found_result else None,
            "errors": worker_errors,
        }
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    output_filename = out_path / f"non_unimodal_result_n{n}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Completed in {elapsed:.2f} seconds.")
    if output_data["errors"]:
        print("!!! ERROR / INCOMPLETE RUN !!!")
        for err in output_data["errors"]:
            print(err)

            
    elif output_data["found"]:
        print("--- First non-unimodal tree found! ---")
        print(f"Tree Index : {output_data['tree_index']}")
        print(f"Graph6     : {output_data['graph6']}")
    else:
        print(
            f"VERIFIED: All {output_data['total_trees_checked']:,} / "
            f"{output_data['expected_trees']:,} trees are unimodal."
        )
    print(f"Saved to: {output_filename}")


if __name__ == "__main__":
    mp.freeze_support()

    start_n = 1
    end_n = 24

    for n_val in range(start_n, end_n + 1):
        print(f"Starting execution for n = {n_val}")
        find_first_non_unimodal(n=n_val, workers=16, output_dir="results")