import json
import argparse

def get_delays(filepath):
    all_requests = []
    traces_raw_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            json_object = json.loads(line.strip())
            traces_raw_data.append(json_object)
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

def train_alpha_model(all_requests):
    processing_times = []
    input_lengths = []
    for request in all_requests:
        processing_times.append(request["e2e_latency"] - (request["queued_time"] + request["prefill_time"] + request["decode_time"]))
        input_lengths.append([request["input_tokens"], 1])
    alpha_model = LinearRegression(positive=True, fit_intercept=False)
    alpha_model.fit(input_lengths, processing_times)

    calculate_metrics(input_lengths, processing_times, alpha_model)

parser = argparse.ArgumentParser(description="Read and parse traces JSON file.")
parser.add_argument("--traces_filepath", help="Path to the vllm traces file to be read.")
args = parser.parse_args()

all_requests = get_delays(args.traces_filepath)
print(len(all_requests))
train_alpha_model(all_requests)