"""Kernels of one steady-state spec step in launch order, with duration and achieved GB/s for GEMVs."""

import sqlite3
import sys
from collections import defaultdict

import numpy as np

db = sys.argv[1] if len(sys.argv) > 1 else r"E:\ABU\FAIE2\bench\profiles\spec_step.sqlite"
con = sqlite3.connect(db)
names = dict(con.execute("SELECT id, value FROM StringIds"))
rows = con.execute("SELECT start, end, demangledName, gridX, gridY FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
starts = [i for i, r in enumerate(rows) if "k_df_noise_tokens" in names[r[2]]]
# the same launch position across many steps -> median duration
steps = [rows[a:b] for a, b in zip(starts[5:], starts[6:])]
L = min(len(s) for s in steps)
med = [np.median([(s[i][1] - s[i][0]) / 1e3 for s in steps]) for i in range(L)]
ref = steps[0]
# bytes per 16-row tile per 256 columns
TILE = {"k_q6k_mma": 3360, "k_q8_mma": 4352}
cols_of = defaultdict(int)
gemv_bytes_known = {}
by = defaultdict(lambda: [0.0, 0.0, 0])
for i in range(L):
    nm = names[ref[i][2]].split("(")[0].replace("void ", "")
    base = nm.split("<")[0]
    print(f"{i:4d} {med[i]:8.1f} us  {base:<20} grid {ref[i][3]}x{ref[i][4]}")
