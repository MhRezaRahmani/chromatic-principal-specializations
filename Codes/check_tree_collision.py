from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sqlite3
import time
from pathlib import Path

import networkx as nx

try:
    from tree_collision_search import (
        PackedPolynomialEngine,
        graph6_text,
        meta_get,
        meta_set,
        save_report,
    )
except ModuleNotFoundError as import_error:
    if import_error.name != "tree_collision_search":
        raise
    import importlib.util

    fallback_path = Path(__file__).with_name(
        "tree_collision_search(1).py"
    )
    if not fallback_path.exists():
        raise
    fallback_spec = importlib.util.spec_from_file_location(
        "tree_collision_search", fallback_path
    )
    fallback_module = importlib.util.module_from_spec(fallback_spec)
    fallback_spec.loader.exec_module(fallback_module)
    PackedPolynomialEngine = fallback_module.PackedPolynomialEngine
    graph6_text = fallback_module.graph6_text
    meta_get = fallback_module.meta_get
    meta_set = fallback_module.meta_set
    save_report = fallback_module.save_report


PROGRAM_VERSION = "PHASE_A_PARENT_ARRAY_ADAPTIVE_ORDERS_PREFIX_V1"
DEFAULT_PRIME = 2305843009213693951  
SUPPORTED_ORDERS = (5, 6, 7, 9, 11, 13)
BENCHMARKED_ORDER_PAIRS = {
    17: (5, 7),
    18: (5, 7),
    19: (7, 11),
    20: (11, 6),
}

_PERIODIC_ENGINE = None
_PACKED_ENGINE = None
_PREFIX_DEGREE = None



def _next_rooted_tree(predecessor, p=None):
    if p is None:
        p = len(predecessor) - 1
        while predecessor[p] == 1:
            p -= 1
    if p == 0:
        return None

    q = p - 1
    while predecessor[q] != predecessor[p] - 1:
        q -= 1

    result = list(predecessor)
    for index in range(p, len(result)):
        result[index] = result[index - p + q]
    return result


def _split_tree(layout):
    one_found = False
    split = None
    for index, level in enumerate(layout):
        if level == 1:
            if one_found:
                split = index
                break
            one_found = True
    if split is None:
        split = len(layout)

    left = [layout[index] - 1 for index in range(1, split)]
    rest = [0] + list(layout[split:])
    return left, rest


def _next_free_tree(candidate):
    left, rest = _split_tree(candidate)
    left_height = max(left)
    rest_height = max(rest)
    valid = rest_height >= left_height
    if valid and rest_height == left_height:
        if len(left) > len(rest):
            valid = False
        elif len(left) == len(rest) and left > rest:
            valid = False

    if valid:
        return candidate

    p = len(left)
    new_candidate = _next_rooted_tree(candidate, p)
    if candidate[p] > 2:
        new_left, _ = _split_tree(new_candidate)
        suffix = range(1, max(new_left) + 2)
        new_candidate[-len(suffix):] = suffix
    return new_candidate


def parent_code_from_levels(levels):
    n = len(levels)
    if n > 255:
        raise ValueError("The compact one-byte parent code supports n <= 255.")
    code = bytearray(n)
    last_at_level = [-1] * (n + 1)
    last_at_level[0] = 0

    for vertex in range(1, n):
        level = levels[vertex]
        parent = last_at_level[level - 1]
        if parent < 0:
            raise AssertionError("Invalid WROM level sequence.")
        code[vertex] = parent + 1
        last_at_level[level] = vertex
    return bytes(code)


def iter_parent_codes(n, max_trees=None):
    if n < 1:
        return
    if n == 1:
        yield 0, bytes((0,))
        return

    layout = list(range(n // 2 + 1)) + list(range(1, (n + 1) // 2))
    tree_index = 0
    while layout is not None:
        layout = _next_free_tree(layout)
        if layout is None:
            break
        yield tree_index, parent_code_from_levels(layout)
        tree_index += 1
        if max_trees is not None and tree_index >= max_trees:
            break
        layout = _next_rooted_tree(layout)


def graph_from_parent_code(code):
    code = bytes(code)
    tree = nx.Graph()
    tree.add_nodes_from(range(len(code)))
    tree.add_edges_from((code[vertex] - 1, vertex)
                        for vertex in range(1, len(code)))
    return tree


def prime_factors(value):
    factors = []
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            factors.append(divisor)
            while value % divisor == 0:
                value //= divisor
        divisor += 1
    if value > 1:
        factors.append(value)
    return factors


def primitive_point(order, prime, preferred_base=3):
    if order < 2 or (prime - 1) % order:
        raise ValueError(f"order {order} must divide prime-1")

    candidates = [preferred_base]
    candidates.extend(base for base in range(2, 100)
                      if base != preferred_base)
    factors = prime_factors(order)
    for base in candidates:
        point = pow(base % prime, (prime - 1) // order, prime)
        if point == 1:
            continue
        if any(pow(point, order // factor, prime) == 1
               for factor in factors):
            continue
        return base, point
    raise ValueError(f"Could not construct a point of order {order}.")


class ParentPeriodicFingerprintEngine:
    def __init__(self, n, order, prime=DEFAULT_PRIME, preferred_base=3):
        self.n = int(n)
        self.order = int(order)
        self.prime = int(prime)
        self.base, self.point = primitive_point(
            self.order, self.prime, preferred_base
        )
        self.weights = [pow(self.point, residue, self.prime)
                        for residue in range(self.order)]
        self.multiplicities = [
            0 if residue >= self.n
            else 1 + (self.n - 1 - residue) // self.order
            for residue in range(self.order)
        ]

    def evaluate_parent_code(self, code):
        code = bytes(code)
        n = self.n
        m = self.order
        p = self.prime
        if len(code) != n:
            raise ValueError("Tree size does not match fingerprint engine.")
        values = self.weights * n
        multiplicities = self.multiplicities

        for vertex in range(n - 1, 0, -1):
            child_offset = vertex * m
            child_total = 0
            for residue in range(m):
                child_total += (
                    multiplicities[residue]
                    * values[child_offset + residue]
                )
            child_total %= p

            parent = code[vertex] - 1
            parent_offset = parent * m
            for residue in range(m):
                values[parent_offset + residue] = (
                    values[parent_offset + residue]
                    * (child_total - values[child_offset + residue])
                ) % p

        result = 0
        for residue in range(m):
            result += multiplicities[residue] * values[residue]
        return result % p


class ParentPackedPolynomialEngine(PackedPolynomialEngine):
    def compute_parent_code(self, code, max_degree=None):
        code = bytes(code)
        n = self.n
        if len(code) != n:
            raise ValueError("Tree size does not match polynomial engine.")

        truncation_mask = self._truncation_mask(max_degree)
        if truncation_mask is None:
            initial = self.qmono
        else:
            initial = [mono & truncation_mask for mono in self.qmono]
        values = [initial.copy() for _ in range(n)]

        for vertex in range(n - 1, 0, -1):
            child_values = values[vertex]
            child_total = sum(child_values)
            if truncation_mask is not None:
                child_total &= truncation_mask

            parent = code[vertex] - 1
            parent_values = values[parent]
            for color in range(n):
                parent_values[color] *= child_total - child_values[color]
                if truncation_mask is not None:
                    parent_values[color] &= truncation_mask
            values[vertex] = None

        result = sum(values[0])
        if truncation_mask is not None:
            result &= truncation_mask
        return result


def integer_to_blob(value):
    length = max(1, (value.bit_length() + 7) // 8)
    return value.to_bytes(length, "little", signed=False)


def blob_to_integer(value):
    return int.from_bytes(bytes(value), "little", signed=False)



def _init_periodic_worker(n, order, prime, preferred_base):
    global _PERIODIC_ENGINE
    _PERIODIC_ENGINE = ParentPeriodicFingerprintEngine(
        n, order, prime, preferred_base
    )


def _periodic_batch(rows):
    return [
        (tree_index, _PERIODIC_ENGINE.evaluate_parent_code(code))
        for tree_index, code in rows
    ]


def _init_prefix_worker(n, degree):
    global _PACKED_ENGINE, _PREFIX_DEGREE
    _PACKED_ENGINE = ParentPackedPolynomialEngine(n)
    _PREFIX_DEGREE = degree


def _prefix_batch(rows):
    output = []
    for tree_index, code in rows:
        packed = _PACKED_ENGINE.compute_parent_code(
            code, max_degree=_PREFIX_DEGREE
        )
        output.append((tree_index, integer_to_blob(packed)))
    return output


def split_rows(rows, batch_size):
    return [rows[start:start + batch_size]
            for start in range(0, len(rows), batch_size)]


def generated_batches(n, batch_size, max_trees, after_index=-1):
    batch = []
    for tree_index, code in iter_parent_codes(n, max_trees):
        if tree_index <= after_index:
            continue
        batch.append((tree_index, code))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch



def choose_adaptive_orders(n):
    if n in BENCHMARKED_ORDER_PAIRS:
        return BENCHMARKED_ORDER_PAIRS[n]

    good = [
        order for order in SUPPORTED_ORDERS
        if n % order not in (0, 1, order - 1)
    ]
    if len(good) < 2:
        good.extend(order for order in SUPPORTED_ORDERS
                    if order not in good)
    return good[0], good[1]


def resolve_orders(n, fp1_order, fp2_order):
    auto1, auto2 = choose_adaptive_orders(n)
    first = auto1 if fp1_order == "auto" else int(fp1_order)
    if fp2_order == "auto":
        choices = [order for order in (auto2,) + SUPPORTED_ORDERS
                   if order != first]
        second = choices[0]
    else:
        second = int(fp2_order)
    if first == second:
        raise ValueError("The two periodic fingerprint orders must differ.")
    return first, second


def open_database(path):
    con = sqlite3.connect(path, timeout=120)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-200000")
    return con


def enforce_metadata(con, settings):
    for key, value in settings.items():
        old = meta_get(con, key)
        if old is not None and old != str(value):
            raise RuntimeError(
                f"Database mismatch for {key}: database has {old}, "
                f"this run asks for {value}. Use a new output directory."
            )
        meta_set(con, key, value)


def initialize_database(path, settings):
    con = open_database(path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS fingerprint_rows (
            tree_index INTEGER PRIMARY KEY,
            fp1 INTEGER NOT NULL
        )
    """)
    enforce_metadata(con, settings)
    con.commit()
    return con


def repeated_group_stats(con, table):
    groups = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    members, pairs = con.execute(f"""
        SELECT COALESCE(SUM(member_count),0),
               COALESCE(SUM(member_count*(member_count-1)/2),0)
        FROM {table}
    """).fetchone()
    return {"groups": int(groups), "members": int(members),
            "pairs": int(pairs)}



def scan_first_fingerprint(
    con, n, order, prime, preferred_base, workers, batch_size,
    max_trees, progress_every,
):
    print("\n=== STAGE 1/5: DIRECT PERIODIC FINGERPRINT ===", flush=True)
    if meta_get(con, "fp1_complete", "0") == "1":
        stored = con.execute(
            "SELECT COUNT(*) FROM fingerprint_rows"
        ).fetchone()[0]
        print(f"Reusing {stored:,} first fingerprints.", flush=True)
        return stored

    last = int(meta_get(con, "fp1_last_processed", "-1"))
    started = time.monotonic()
    computed = 0
    next_progress = progress_every

    batches = generated_batches(n, batch_size, max_trees, last)
    if workers > 1:
        pool = mp.Pool(
            workers,
            initializer=_init_periodic_worker,
            initargs=(n, order, prime, preferred_base),
        )
        results = pool.imap(_periodic_batch, batches, chunksize=1)
    else:
        _init_periodic_worker(n, order, prime, preferred_base)
        pool = None
        results = map(_periodic_batch, batches)

    try:
        for rows in results:
            con.executemany(
                "INSERT OR REPLACE INTO fingerprint_rows(tree_index,fp1) "
                "VALUES(?,?)",
                rows,
            )
            last = rows[-1][0]
            computed += len(rows)
            meta_set(con, "fp1_last_processed", last)
            con.commit()
            if computed >= next_progress:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"  first fingerprints: {computed:,} this run "
                    f"({computed/elapsed:,.1f} trees/s)",
                    flush=True,
                )
                next_progress += progress_every
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    stored = con.execute(
        "SELECT COUNT(*) FROM fingerprint_rows"
    ).fetchone()[0]
    meta_set(con, "fp1_complete", 1)
    meta_set(con, "tree_count", stored)
    con.commit()

    if max_trees is None:
        expected = nx.number_of_nonisomorphic_trees(n)
        if stored != expected:
            raise AssertionError(
                f"Generated {stored} trees, expected {expected}."
            )
    print(f"First pass complete: {stored:,} trees.", flush=True)
    return stored


def build_fp1_candidates(con, total):
    print("\n=== STAGE 2/5: FIRST-FINGERPRINT GROUPING ===", flush=True)
    if meta_get(con, "fp1_groups_ready", "0") == "1":
        return {
            name: int(meta_get(con, f"fp1_repeated_{name}", "0"))
            for name in ("groups", "members", "pairs")
        }

    con.execute("DROP TABLE IF EXISTS fp1_groups")
    con.execute("DROP TABLE IF EXISTS fp1_candidates")
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_fingerprint_rows_fp1 "
        "ON fingerprint_rows(fp1)"
    )
    con.execute("""
        CREATE TABLE fp1_groups(
            fp1 INTEGER PRIMARY KEY,
            member_count INTEGER NOT NULL
        ) WITHOUT ROWID
    """)
    con.execute("""
        INSERT INTO fp1_groups
        SELECT fp1,COUNT(*) FROM fingerprint_rows
        GROUP BY fp1 HAVING COUNT(*)>=2
    """)
    con.execute("""
        CREATE TABLE fp1_candidates(
            tree_index INTEGER PRIMARY KEY,
            fp1 INTEGER NOT NULL,
            fp2 INTEGER,
            tree_code BLOB
        )
    """)
    con.execute("""
        INSERT INTO fp1_candidates(tree_index,fp1)
        SELECT r.tree_index,r.fp1
        FROM fingerprint_rows r JOIN fp1_groups g ON r.fp1=g.fp1
    """)
    con.commit()
    stats = repeated_group_stats(con, "fp1_groups")
    for name, value in stats.items():
        meta_set(con, f"fp1_repeated_{name}", value)
    meta_set(con, "fp1_groups_ready", 1)
    con.commit()
    print(
        f"First fingerprint: {total-stats['members']:,} singletons; "
        f"{stats['groups']:,} repeated groups containing "
        f"{stats['members']:,} trees.",
        flush=True,
    )
    return stats


def materialize_candidate_codes(con, n, max_trees):
    missing = {
        row[0] for row in con.execute(
            "SELECT tree_index FROM fp1_candidates WHERE tree_code IS NULL"
        )
    }
    if not missing:
        return
    print(
        f"Regenerating compact codes for {len(missing):,} survivors.",
        flush=True,
    )
    updates = []
    for tree_index, code in iter_parent_codes(n, max_trees):
        if tree_index not in missing:
            continue
        updates.append((sqlite3.Binary(code), tree_index))
        missing.remove(tree_index)
        if len(updates) >= 5000:
            con.executemany(
                "UPDATE fp1_candidates SET tree_code=? WHERE tree_index=?",
                updates,
            )
            con.commit()
            updates = []
        if not missing:
            break
    if updates:
        con.executemany(
            "UPDATE fp1_candidates SET tree_code=? WHERE tree_index=?",
            updates,
        )
        con.commit()
    if missing:
        raise RuntimeError(f"Could not regenerate {len(missing)} trees.")


def compute_second_fingerprints(
    con, n, order, prime, preferred_base, workers, batch_size, candidate_count,
):
    print("\n=== STAGE 3/5: SECOND PERIODIC FINGERPRINT ===", flush=True)
    if candidate_count == 0:
        print("No first-pass collisions; stage skipped.", flush=True)
        meta_set(con, "fp2_complete", 1)
        con.commit()
        return
    if meta_get(con, "fp2_complete", "0") == "1":
        print("Reusing second fingerprints.", flush=True)
        return

    if workers > 1:
        pool = mp.Pool(
            workers,
            initializer=_init_periodic_worker,
            initargs=(n, order, prime, preferred_base),
        )
    else:
        _init_periodic_worker(n, order, prime, preferred_base)
        pool = None

    processed = 0
    wave = batch_size * max(1, workers)
    try:
        while True:
            rows = con.execute("""
                SELECT tree_index,tree_code FROM fp1_candidates
                WHERE fp2 IS NULL ORDER BY tree_index LIMIT ?
            """, (wave,)).fetchall()
            if not rows:
                break
            tasks = split_rows(rows, batch_size)
            results = (map(_periodic_batch, tasks) if pool is None
                       else pool.map(_periodic_batch, tasks))
            updates = []
            for result in results:
                updates.extend((value, index) for index, value in result)
            con.executemany(
                "UPDATE fp1_candidates SET fp2=? WHERE tree_index=?",
                updates,
            )
            con.commit()
            processed += len(updates)
            if processed % max(wave * 10, 1) == 0 or processed == candidate_count:
                print(f"  second fingerprints: {processed:,}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    meta_set(con, "fp2_complete", 1)
    con.commit()


def build_fp2_candidates(con, fp1_members):
    if meta_get(con, "fp2_groups_ready", "0") == "1":
        return {
            name: int(meta_get(con, f"fp2_repeated_{name}", "0"))
            for name in ("groups", "members", "pairs")
        }
    con.execute("DROP TABLE IF EXISTS fp2_groups")
    con.execute("DROP TABLE IF EXISTS fp2_candidates")
    con.execute("""
        CREATE TABLE fp2_groups(
            fp1 INTEGER NOT NULL,
            fp2 INTEGER NOT NULL,
            member_count INTEGER NOT NULL,
            PRIMARY KEY(fp1,fp2)
        ) WITHOUT ROWID
    """)
    con.execute("""
        INSERT INTO fp2_groups
        SELECT fp1,fp2,COUNT(*) FROM fp1_candidates
        GROUP BY fp1,fp2 HAVING COUNT(*)>=2
    """)
    con.execute("""
        CREATE TABLE fp2_candidates(
            tree_index INTEGER PRIMARY KEY,
            fp1 INTEGER NOT NULL,
            fp2 INTEGER NOT NULL,
            prefix BLOB,
            tree_code BLOB NOT NULL
        )
    """)
    con.execute("""
        INSERT INTO fp2_candidates(tree_index,fp1,fp2,tree_code)
        SELECT r.tree_index,r.fp1,r.fp2,r.tree_code
        FROM fp1_candidates r JOIN fp2_groups g
          ON r.fp1=g.fp1 AND r.fp2=g.fp2
    """)
    con.commit()
    stats = repeated_group_stats(con, "fp2_groups")
    for name, value in stats.items():
        meta_set(con, f"fp2_repeated_{name}", value)
    meta_set(con, "fp2_groups_ready", 1)
    con.commit()
    print(
        f"Second fingerprint: {fp1_members-stats['members']:,} additional "
        f"singletons; {stats['groups']:,} groups containing "
        f"{stats['members']:,} trees.",
        flush=True,
    )
    return stats


def compute_exact_prefixes(
    con, n, degree, workers, batch_size, candidate_count,
):
    print(f"\n=== STAGE 4/5: EXACT PREFIX THROUGH q^{degree} ===", flush=True)
    if candidate_count == 0:
        print("No double-fingerprint collisions; stage skipped.", flush=True)
        meta_set(con, "prefix_complete", 1)
        con.commit()
        return
    if meta_get(con, "prefix_complete", "0") == "1":
        print("Reusing exact prefixes.", flush=True)
        return

    if workers > 1:
        pool = mp.Pool(
            workers,
            initializer=_init_prefix_worker,
            initargs=(n, degree),
        )
    else:
        _init_prefix_worker(n, degree)
        pool = None
    processed = 0
    wave = batch_size * max(1, workers)
    try:
        while True:
            rows = con.execute("""
                SELECT tree_index,tree_code FROM fp2_candidates
                WHERE prefix IS NULL ORDER BY tree_index LIMIT ?
            """, (wave,)).fetchall()
            if not rows:
                break
            tasks = split_rows(rows, batch_size)
            results = (map(_prefix_batch, tasks) if pool is None
                       else pool.map(_prefix_batch, tasks))
            updates = []
            for result in results:
                updates.extend((sqlite3.Binary(blob), index)
                               for index, blob in result)
            con.executemany(
                "UPDATE fp2_candidates SET prefix=? WHERE tree_index=?",
                updates,
            )
            con.commit()
            processed += len(updates)
            print(f"  exact prefixes: {processed:,}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    meta_set(con, "prefix_complete", 1)
    con.commit()


def build_prefix_candidates(con, fp2_members):
    if meta_get(con, "prefix_groups_ready", "0") == "1":
        return {
            name: int(meta_get(con, f"prefix_repeated_{name}", "0"))
            for name in ("groups", "members", "pairs")
        }
    con.execute("DROP TABLE IF EXISTS prefix_groups")
    con.execute("DROP TABLE IF EXISTS final_candidates")
    con.execute("""
        CREATE TABLE prefix_groups(
            fp1 INTEGER NOT NULL,
            fp2 INTEGER NOT NULL,
            prefix BLOB NOT NULL,
            member_count INTEGER NOT NULL,
            PRIMARY KEY(fp1,fp2,prefix)
        ) WITHOUT ROWID
    """)
    con.execute("""
        INSERT INTO prefix_groups
        SELECT fp1,fp2,prefix,COUNT(*) FROM fp2_candidates
        GROUP BY fp1,fp2,prefix HAVING COUNT(*)>=2
    """)
    con.execute("""
        CREATE TABLE final_candidates(
            tree_index INTEGER PRIMARY KEY,
            fp1 INTEGER NOT NULL,
            fp2 INTEGER NOT NULL,
            prefix BLOB NOT NULL,
            tree_code BLOB NOT NULL,
            full_processed INTEGER NOT NULL DEFAULT 0
        )
    """)
    con.execute("""
        INSERT INTO final_candidates(tree_index,fp1,fp2,prefix,tree_code)
        SELECT r.tree_index,r.fp1,r.fp2,r.prefix,r.tree_code
        FROM fp2_candidates r JOIN prefix_groups g
          ON r.fp1=g.fp1 AND r.fp2=g.fp2 AND r.prefix=g.prefix
    """)
    con.commit()
    stats = repeated_group_stats(con, "prefix_groups")
    for name, value in stats.items():
        meta_set(con, f"prefix_repeated_{name}", value)
    meta_set(con, "prefix_groups_ready", 1)
    con.commit()
    print(
        f"Exact prefix: {fp2_members-stats['members']:,} additional "
        f"singletons; {stats['groups']:,} final groups containing "
        f"{stats['members']:,} trees.",
        flush=True,
    )
    return stats


def initialize_full_tables(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS full_seen(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fp1 INTEGER NOT NULL,
            fp2 INTEGER NOT NULL,
            prefix BLOB NOT NULL,
            digest BLOB NOT NULL,
            tree_code BLOB NOT NULL,
            tree_index INTEGER NOT NULL
        )
    """)
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_full_seen_lookup
        ON full_seen(fp1,fp2,prefix,digest)
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS collisions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tree_index_a INTEGER NOT NULL,
            tree_index_b INTEGER NOT NULL,
            fp1 INTEGER NOT NULL,
            fp2 INTEGER NOT NULL,
            prefix BLOB NOT NULL,
            digest BLOB NOT NULL,
            tree_code_a BLOB NOT NULL,
            tree_code_b BLOB NOT NULL,
            UNIQUE(tree_index_a,tree_index_b)
        )
    """)
    con.commit()


def compute_full_polynomials(
    con, n, degree, run_dir, candidate_count, commit_every,
    stop_on_collision,
):
    print("\n=== STAGE 5/5: COMPLETE EXACT VERIFICATION ===", flush=True)
    initialize_full_tables(con)
    if candidate_count == 0:
        print("No trees require complete polynomials.", flush=True)
        meta_set(con, "full_complete", 1)
        con.commit()
        return 0

    engine = ParentPackedPolynomialEngine(n)
    prefix_mask = engine._truncation_mask(degree)
    processed = 0
    rows = con.execute("""
        SELECT tree_index,fp1,fp2,prefix,tree_code
        FROM final_candidates WHERE full_processed=0 ORDER BY tree_index
    """)
    for tree_index, fp1, fp2, prefix_blob, code in rows:
        code = bytes(code)
        packed = engine.compute_parent_code(code)
        expected_prefix = blob_to_integer(prefix_blob)
        if packed & prefix_mask != expected_prefix:
            raise AssertionError(
                f"Exact prefix mismatch at tree index {tree_index}."
            )
        raw = integer_to_blob(packed)
        digest = hashlib.blake2b(raw, digest_size=32).digest()

        prior = con.execute("""
            SELECT tree_index,tree_code FROM full_seen
            WHERE fp1=? AND fp2=? AND prefix=? AND digest=?
        """, (
            fp1, fp2, sqlite3.Binary(bytes(prefix_blob)),
            sqlite3.Binary(digest),
        )).fetchall()

        collision_found = False
        for old_index, old_code in prior:
            old_packed = engine.compute_parent_code(old_code)
            if old_packed != packed:
                continue
            collision_found = True
            index_a, index_b = sorted((old_index, tree_index))
            code_a, code_b = ((bytes(old_code), code)
                              if old_index <= tree_index
                              else (code, bytes(old_code)))
            con.execute("""
                INSERT OR IGNORE INTO collisions(
                    tree_index_a,tree_index_b,fp1,fp2,prefix,digest,
                    tree_code_a,tree_code_b
                ) VALUES(?,?,?,?,?,?,?,?)
            """, (
                index_a, index_b, fp1, fp2,
                sqlite3.Binary(bytes(prefix_blob)), sqlite3.Binary(digest),
                sqlite3.Binary(code_a), sqlite3.Binary(code_b),
            ))
            tree_a = graph_from_parent_code(code_a)
            tree_b = graph_from_parent_code(code_b)
            save_report(
                run_dir / f"collision_{n}_{index_a}_{index_b}.json",
                n,
                f"fp1={fp1}|fp2={fp2}|prefix_through_{degree}",
                index_a,
                index_b,
                graph6_text(tree_a),
                graph6_text(tree_b),
                engine,
                packed,
            )
            print(
                f"!!! EXACT COLLISION: indices {index_a}, {index_b}",
                flush=True,
            )
            break

        con.execute("""
            INSERT INTO full_seen(fp1,fp2,prefix,digest,tree_code,tree_index)
            VALUES(?,?,?,?,?,?)
        """, (
            fp1, fp2, sqlite3.Binary(bytes(prefix_blob)),
            sqlite3.Binary(digest), sqlite3.Binary(code), tree_index,
        ))
        con.execute(
            "UPDATE final_candidates SET full_processed=1 WHERE tree_index=?",
            (tree_index,),
        )
        processed += 1
        if processed % commit_every == 0 or collision_found:
            con.commit()
        if collision_found and stop_on_collision:
            break

    con.commit()
    remaining = con.execute(
        "SELECT COUNT(*) FROM final_candidates WHERE full_processed=0"
    ).fetchone()[0]
    if remaining == 0:
        meta_set(con, "full_complete", 1)
        con.commit()
    return con.execute("SELECT COUNT(*) FROM collisions").fetchone()[0]


def self_test():

    def reference_periodic_graph(tree, engine):
        n = engine.n
        m = engine.order
        p = engine.prime
        parent = [-2] * n
        parent[0] = -1
        children = [[] for _ in range(n)]
        traversal = [0]
        for vertex in traversal:
            for neighbor in tree[vertex]:
                if neighbor == parent[vertex]:
                    continue
                parent[neighbor] = vertex
                children[vertex].append(neighbor)
                traversal.append(neighbor)

        values = [None] * n
        for vertex in reversed(traversal):
            current = engine.weights.copy()
            for child in children[vertex]:
                child_values = values[child]
                child_total = sum(
                    engine.multiplicities[residue]
                    * child_values[residue]
                    for residue in range(m)
                ) % p
                for residue in range(m):
                    current[residue] = (
                        current[residue]
                        * (child_total - child_values[residue])
                    ) % p
            values[vertex] = current
        return sum(
            engine.multiplicities[residue] * values[0][residue]
            for residue in range(m)
        ) % p

    checked = 0
    for n in range(2, 10):
        records = list(iter_parent_codes(n))
        graphs = list(nx.nonisomorphic_trees(n))
        assert len(records) == len(graphs)

        order = choose_adaptive_orders(n)[0]
        direct_mod = ParentPeriodicFingerprintEngine(n, order)
        direct_poly = ParentPackedPolynomialEngine(n)
        old_poly = PackedPolynomialEngine(n)
        degree = min(2 * n, n * (n - 1))

        for (tree_index, code), graph in zip(records, graphs):
            rebuilt = graph_from_parent_code(code)
            assert {
                frozenset(edge) for edge in rebuilt.edges()
            } == {
                frozenset(edge) for edge in graph.edges()
            }
            assert direct_mod.evaluate_parent_code(code) == (
                reference_periodic_graph(graph, direct_mod)
            )
            assert direct_poly.compute_parent_code(
                code, degree
            ) == old_poly.compute(graph, degree)
            assert direct_poly.compute_parent_code(code) == old_poly.compute(graph)
            checked += 1
    print(f"SELF-TEST PASSED: {checked:,} trees checked.")


def run_phase_a(
    n=18,
    coefficient_degree=None,
    batch_size=1000,
    workers=None,
    shards=4,
    output_dir=None,
    fp1_order="auto",
    fp2_order="auto",
    prime=DEFAULT_PRIME,
    preferred_base=3,
    max_trees=None,
    progress_every=100000,
    stop_on_collision=True,
):
    if n < 2:
        raise ValueError("n must be at least 2")
    if workers is None:
        workers = mp.cpu_count()
    if workers < 1 or batch_size < 1 or shards < 1:
        raise ValueError("workers, shards and batch_size must be positive")
    max_degree = n * (n - 1)
    if coefficient_degree is None:
        coefficient_degree = min(2 * n, max_degree)
    if not 0 <= coefficient_degree <= max_degree:
        raise ValueError(f"coefficient degree must lie in [0,{max_degree}]")

    order1, order2 = resolve_orders(n, fp1_order, fp2_order)
    engine1 = ParentPeriodicFingerprintEngine(
        n, order1, prime, preferred_base
    )
    engine2 = ParentPeriodicFingerprintEngine(
        n, order2, prime, preferred_base
    )

    root_dir = Path(output_dir or f"phase_a_search_n{n}")
    run_dir = root_dir / (
        f"orders{order1}-{order2}_prefix{coefficient_degree}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = run_dir / "search.sqlite"
    settings = {
        "program_version": PROGRAM_VERSION,
        "n": n,
        "coefficient_degree": coefficient_degree,
        "fp1_order": order1,
        "fp1_base": engine1.base,
        "fp1_point": engine1.point,
        "fp2_order": order2,
        "fp2_base": engine2.base,
        "fp2_point": engine2.point,
        "prime": prime,
        "max_trees": "all" if max_trees is None else max_trees,
    }
    con = initialize_database(db_path, settings)
    started = time.monotonic()
    try:
        print("Starting Phase-A collision search", flush=True)
        print(f"n                       = {n}", flush=True)
        print(
            f"periodic fingerprints   = orders {order1} then {order2}",
            flush=True,
        )
        print(
            f"exact fallback          = complete prefix through "
            f"q^{coefficient_degree}",
            flush=True,
        )
        print(f"workers                  = {workers}", flush=True)
        print(
            f"requested shards         = {shards} "
            "(single enumeration; no duplicated shard scans)",
            flush=True,
        )
        print(f"output                   = {run_dir}", flush=True)

        total = scan_first_fingerprint(
            con, n, order1, prime, preferred_base, workers, batch_size,
            max_trees, progress_every,
        )
        fp1 = build_fp1_candidates(con, total)
        materialize_candidate_codes(con, n, max_trees)
        compute_second_fingerprints(
            con, n, order2, prime, preferred_base, workers, batch_size,
            fp1["members"],
        )
        fp2 = build_fp2_candidates(con, fp1["members"])
        compute_exact_prefixes(
            con, n, coefficient_degree, workers, batch_size, fp2["members"]
        )
        prefix = build_prefix_candidates(con, fp2["members"])
        collisions = compute_full_polynomials(
            con, n, coefficient_degree, run_dir, prefix["members"],
            max(1, batch_size), stop_on_collision,
        )
        elapsed = time.monotonic() - started

        summary = {
            "program_version": PROGRAM_VERSION,
            "n": n,
            "tree_count": total,
            "workers": workers,
            "requested_shards": shards,
            "order1": order1,
            "order2": order2,
            "fp1_repeated_members": fp1["members"],
            "fp2_repeated_members": fp2["members"],
            "prefix_repeated_members": prefix["members"],
            "exact_collisions": collisions,
            "wall_seconds_internal": elapsed,
        }
        with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

        print("\n" + "=" * 72)
        print("PHASE-A SEARCH SUMMARY")
        print("=" * 72)
        print(f"all generated trees              : {total:,}")
        print(f"trees receiving order {order2:<2}          : {fp1['members']:,}")
        print(f"trees receiving exact prefix    : {fp2['members']:,}")
        print(f"trees receiving full polynomial : {prefix['members']:,}")
        print(f"exact polynomial collisions     : {collisions:,}")
        print(f"internal wall time              : {elapsed:.3f} s")
        print(f"database                        : {db_path}")
        if max_trees is not None:
            print("WARNING: partial --max-trees run.")
        print("=" * 72)
        return 0
    finally:
        con.close()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Phase-A adaptive parent-array collision search."
    )
    parser.add_argument("--n", type=int, default=18)
    parser.add_argument("--coefficient-degree", type=int)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument(
        "--shards", type=int, default=4,
        help=(
            "accepted for command compatibility; Phase A enumerates once "
            "and distributes batches instead of duplicating shard scans"
        ),
    )
    parser.add_argument("--fp1-order", default="auto")
    parser.add_argument("--fp2-order", default="auto")
    parser.add_argument("--prime", type=int, default=DEFAULT_PRIME)
    parser.add_argument("--preferred-base", type=int, default=3)
    parser.add_argument("--max-trees", type=int)
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--bucket-mode", choices=("none",), default="none",
        help="retained for compatibility; unsafe structural pruning is disabled",
    )
    parser.add_argument(
        "--stop-on-collision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--self-test", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test()
        return 0
    return run_phase_a(
        n=args.n,
        coefficient_degree=args.coefficient_degree,
        batch_size=args.batch_size,
        workers=args.workers,
        shards=args.shards,
        output_dir=args.output_dir,
        fp1_order=args.fp1_order,
        fp2_order=args.fp2_order,
        prime=args.prime,
        preferred_base=args.preferred_base,
        max_trees=args.max_trees,
        progress_every=args.progress_every,
        stop_on_collision=args.stop_on_collision,
    )


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
