import argparse, json
from pathlib import Path

def read(path):
    data = json.loads(Path(path).read_text())
    required = {"simulation_count", "runtime_seconds", "var_95", "var_99", "seed"}
    if required - data.keys():
        raise ValueError(f"{path} missing {sorted(required-data.keys())}")
    return data

p = argparse.ArgumentParser()
p.add_argument("your_result")
p.add_argument("other_result")
a = p.parse_args()
x, y = read(a.your_result), read(a.other_result)
for k in ("simulation_count", "seed"):
    print(f"{k}: {x[k]} vs {y[k]} {'MATCH' if x[k] == y[k] else 'MISMATCH'}")
for k in ("runtime_seconds", "var_95", "var_99"):
    print(f"{k}: yours={x[k]} | other={y[k]}")
if x["runtime_seconds"] > 0 and y["runtime_seconds"] > 0:
    print(f"Runtime ratio (yours/other): {x['runtime_seconds']/y['runtime_seconds']:.3f}x")
print("Runtime is comparable only on the same machine and software environment.")
