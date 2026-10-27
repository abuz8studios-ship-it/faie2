"""Achieved bandwidth of each k_q6k_mma / k_q8_mma launch shape in a spec_step capture (grid.x = rows/16)."""

import sqlite3
import sys
from collections import defaultdict

import numpy as np

db = sys.argv[1] if len(sys.argv) > 1 else r"E:\ABU\FAIE2\bench\profiles\spec_step.sqlite"
con = sqlite3.connect(db)
names = dict(con.execute("SELECT id, value FROM StringIds"))
rows = con.execute("SELECT start, end, demangledName, gridX FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
first_step = next(i for i, r in enumerate(rows) if "k_df_noise_tokens" in names[r[2]])
# rows/16 -> (cols, bytes per 16x256 tile) for every shape the forward and drafter use
SHAPES = {
    "q6": {16480 // 16: ("recurrent in_proj", 5120), 6144 // 16: None, 34816 // 16: ("gate_up", 5120),
           5120 // 16: None, 14336 // 16: ("attn in_proj", 5120), 248320 // 16: ("lm_head", 5120)},
}
d = defaultdict(list)
for s, e, n, gx in rows[first_step:]:
    nm = names[n]
    kind = "q6" if "k_q6k_mma" in nm else "q8" if "k_q8_mma" in nm else None
    if kind:
        d[(kind, gx)].append((e - s) / 1e3)
for (kind, gx), v in sorted(d.items()):
    print(f"{kind} rows {gx * 16:7d}  launches {len(v):6d}  median {np.median(v):8.1f} us  min {np.min(v):8.1f} us")
