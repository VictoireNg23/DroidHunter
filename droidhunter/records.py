"""Compact per-APK instruction records.

Each APK is stored as one .npz: three small string tables (opcodes, op1, op2 seen in
that APK) and three int32 index arrays of length n_instructions. Reading a record is a
single np.load, and counting or encoding is vectorised instead of string-by-string.
"""
from pathlib import Path

import numpy as np

FIELDS = ("op", "reg", "arg")


class RecordBuilder:
    def __init__(self):
        self.tables = [dict(), dict(), dict()]
        self.index = [[], [], []]

    def add(self, triple):
        for table, idx, value in zip(self.tables, self.index, triple):
            i = table.get(value)
            if i is None:
                i = table[value] = len(table)
            idx.append(i)

    def __len__(self):
        return len(self.index[0])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for name, table, idx in zip(FIELDS, self.tables, self.index):
            arrays[f"{name}_table"] = np.array(list(table), dtype=str)
            arrays[f"{name}_idx"] = np.asarray(idx, dtype=np.int32)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)


class Record:
    """Loaded record: tables[k] (array of str) and idx[k] (int32 array)."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.tables = [z[f"{n}_table"] for n in FIELDS]
            self.idx = [z[f"{n}_idx"] for n in FIELDS]

    def __len__(self):
        return len(self.idx[0])

    def counts(self, k):
        """Occurrences of each local string of field k, aligned with tables[k]."""
        return np.bincount(self.idx[k], minlength=len(self.tables[k]))
