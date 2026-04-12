"""Quick wrapper to run benchmarks and capture results to a clean UTF-8 file."""
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('PYTHONPATH', '.')

# Capture all stdout to a file
import io
output = io.StringIO()
old_stdout = sys.stdout

# Run the benchmark
from benchmarks.runner import main as runner_main
sys.argv = ['runner', '--limit', '2', '--mode', 'baseline,balanced', '--mock']

# Tee output to both console and buffer
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = Tee(old_stdout, output)

try:
    runner_main()
except SystemExit:
    pass

sys.stdout = old_stdout

# Write clean results
with open('bench_clean_results.txt', 'w', encoding='utf-8') as f:
    f.write(output.getvalue())

print("\n=== Results saved to bench_clean_results.txt ===")
