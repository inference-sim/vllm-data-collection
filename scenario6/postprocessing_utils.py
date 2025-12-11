import re
import subprocess

def run_go_binary(arguments, go_binary_path, metrics_lock = None):
    result = subprocess.run(
        [go_binary_path] + arguments,
        capture_output=True,
        text=True,
        check=True,
        encoding='utf-8'
    )

    if not metrics_lock:
        if result.stderr:
            print(
                f"Go binary error output:\n{result.stderr}")
        sim_metrics = parse_sim_metrics_to_json(result.stdout)
        return sim_metrics
    else:
        with metrics_lock:
            if result.stderr:
                print(
                    f"Go binary error output:\n{result.stderr}")
            sim_metrics = parse_sim_metrics_to_json(result.stdout)
            return sim_metrics
    
def parse_sim_metrics_to_json(stdout):
    """
    Reads text from standard input, parses key-value metrics,
    and prints a single JSON object to standard output.
    """

    metrics_data = {}
    metric_pattern = re.compile(r'^\s*(.+?)\s*:\s*([\d\.]+)')

    for line in stdout.split('\n'):
        match = metric_pattern.search(line)
        if match:
            key = match.group(1).strip()
            key = key.rstrip(":")
            value_str = match.group(2)

            try:
                if '.' in value_str:
                    value = float(value_str)
                else:
                    value = int(value_str)
                metrics_data[key] = value
            except ValueError:
                continue
    return metrics_data

