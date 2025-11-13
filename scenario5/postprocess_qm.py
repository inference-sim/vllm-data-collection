import argparse
import json
import pandas as pd

def read_traces_jsonl(filepath):
    traces_raw_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            json_object = json.loads(line.strip())
            traces_raw_data.append(json_object)
    return traces_raw_data

def get_server_side_info(traces_raw_data):
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
                            request["request_id"] = attribute["value"]["stringValue"]
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

def get_average_metrics_per_benchmark(benchmark_df, request_rate):
    benchmark_df['ITL'] = benchmark_df['decode_time'] / benchmark_df['output_tokens']
    all_means = benchmark_df[['input_tokens', 'output_tokens', 'queued_time', 'prefill_time', 'ITL']].mean()
    benchmark_averages = {
        "requestRate":    request_rate,
        "inputTokens":    all_means['input_tokens'],
        "outputTokens":   all_means['output_tokens'],
        "avgWaitTime":    all_means['queued_time'],
        "avgPrefillTime": all_means['prefill_time'],
        "avgITLTime":     all_means['ITL'],
    }
    return benchmark_averages

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Read and parse traces JSON file.")
    parser.add_argument("--traces_filepath", 
                        help="Path to the vllm traces file to be read.")
    parser.add_argument("--sweep_info_path",
                        default="sweep_info.json",
                        help="Path to GuideLLM's sweep trial info")
    args = parser.parse_args()

    traces_raw_data = read_traces_jsonl(args.traces_filepath)
    all_requests = get_server_side_info(traces_raw_data)
    requests_df = pd.DataFrame(all_requests)

    # read GuideLLM sweep info
    with open(args.sweep_info_path, 'r') as f:
        sweep_info = json.load(f)

    qm_training_data = []
    for sweep in sweep_info:
        # each request-rate forms a new benchmark
        rps = sweep["rps"]
        benchmark_request_ids = sweep["requestIDs"]
        benchmark_df = requests_df[requests_df["request_id"].isin(benchmark_request_ids)].copy()
        benchmark_averages = get_average_metrics_per_benchmark(benchmark_df, rps)
        qm_training_data.append(benchmark_averages)
    print(qm_training_data)