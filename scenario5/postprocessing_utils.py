import json
import re
import subprocess

BLIS_REQGEN_CONFIG_FOLDER = "blis_reqgenconfigs"
SWEEP_INFO_FILENAME = "sweep_info.json"
QM_TRAINING_FILEPATH = "QM_train.json"
QM_TESTING_FILEPATH = "QM_test.json"
BLIS_TRAINING_FILEPATH = "BLIS_train.json"
BLIS_TESTING_FILEPATH = "BLIS_test.json"
ALPHA_WEIGHTS_FILENAME = 'BLIS_alpha_weights.pkl'
ALPHA_METRICS_FILENAME = 'BLIS_alpha_metrics.json'
BETA_METRICS_FILENAME = 'BLIS_beta_metrics.json'

def read_traces_jsonl(filepath):
    """
    vLLM traces is a JSONL file.
    This function loads raw data from traces file.
    """
    traces_raw_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            json_object = json.loads(line.strip())
            traces_raw_data.append(json_object)
    return traces_raw_data

def get_server_side_metrics_from_traces(traces_raw_data):
    """
    Fetch server-side info from traces file per request.
    We currently fetch vLLM requestID, input, output tokens, 
    e2e latency, waiting, prefill and decode time per request.
    """
    all_requests = []
    for data in traces_raw_data:
        for resourceSpan in data["resourceSpans"]:
            for scopeSpan in resourceSpan["scopeSpans"]:
                for span in scopeSpan["spans"]:
                    request = {}
                    for attribute in span["attributes"]:
                        if attribute["key"] == "gen_ai.request.id":
                            request["request_id"] = attribute["value"]["stringValue"].rsplit("-0", 1)[0]
                        if attribute["key"] == "gen_ai.usage.prompt_tokens":
                            request["input_tokens"] = int(attribute["value"]["intValue"])
                        if attribute["key"] == "gen_ai.usage.completion_tokens":
                            request["output_tokens"] = int(attribute["value"]["intValue"])
                        if attribute["key"] == "gen_ai.latency.e2e":
                            request["e2e_latency"] = attribute["value"]["doubleValue"]
                        if attribute["key"] == "gen_ai.latency.time_in_queue":
                            request["queued_time"] = attribute["value"]["doubleValue"]
                        if attribute["key"] == "gen_ai.latency.time_in_model_prefill":
                            request["prefill_time"] = attribute["value"]["doubleValue"]
                        if attribute["key"] == "gen_ai.latency.time_in_model_decode":
                            request["decode_time"] = attribute["value"]["doubleValue"]
                    all_requests.append(request)
    return all_requests

def run_go_binary(arguments, go_binary_path, request_rate, metrics_lock = None):
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
        sim_metrics = parse_sim_metrics_to_json(result.stdout, request_rate)
        return sim_metrics
    else:
        with metrics_lock:
            if result.stderr:
                print(
                    f"Go binary error output:\n{result.stderr}")
            sim_metrics = parse_sim_metrics_to_json(result.stdout, request_rate)
            return sim_metrics
    
def parse_sim_metrics_to_json(stdout, request_rate):
    """
    Reads text from standard input, parses key-value metrics,
    and prints a single JSON object to standard output.
    """

    metrics_data = {}
    metric_pattern = re.compile(r'^\s*(.+?)\s*:\s*([\d\.]+)')
    metrics_data["request_rate"] = request_rate

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
