import argparse, json
from pathlib import Path
from src.benchmarks import benchmark_engine

parser = argparse.ArgumentParser()
parser.add_argument("--count", type=int, default=50000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", default="results/benchmark.json")
args = parser.parse_args()
result = benchmark_engine(args.count, args.seed)
path = Path(args.output)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
