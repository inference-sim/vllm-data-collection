import glob
import json
import os
import csv
import sys

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

# Helper to get values from the nested OTLP value structure
def get_val(attrs, key, default=0):
    val_obj = attrs.get(key, {})
    return val_obj.get("stringValue") or val_obj.get("doubleValue") or val_obj.get("intValue") or default

def find_global_min_start(input_file, requestIDs):
    traces_data = read_traces_jsonl(input_file)
    global_start = float("Inf")

    # Navigate the OTLP JSON structure
    for trace in traces_data:
        for resource_span in trace.get("resourceSpans", []):
            for scope_span in resource_span.get("scopeSpans", []):
                for span in scope_span.get("spans", []):
                    
                    # Convert Unix Nano to Seconds for the CSV
                    start_time = int(span["startTimeUnixNano"]) / 1e9
                    
                    # Extract attributes into a easy-to-access dictionary
                    attrs = {attr["key"]: attr["value"] for attr in span.get("attributes", [])}

                    request_id = get_val(attrs, "gen_ai.request.id", "unknown").rsplit("-0", 1)[0]
                    if request_id in requestIDs:
                        prefill_start = start_time + float(get_val(attrs, "gen_ai.latency.time_in_queue", 0))
                        global_start = min(global_start, prefill_start)
    return global_start


def process_traces(input_file, output_file, requestIDs, global_min_start_time):
    traces_data = read_traces_jsonl(input_file)

    rows = []

    # Navigate the OTLP JSON structure
    for data in traces_data:
        for resourceSpan in data["resourceSpans"]:
            for scopeSpan in resourceSpan["scopeSpans"]:
                for span in scopeSpan["spans"]:
                    # Convert Unix Nano to Seconds for the CSV
                    start_time = int(span["startTimeUnixNano"]) / 1e9
                    attrs = {attr["key"]: attr["value"] for attr in span.get("attributes", [])}
                    request_id = get_val(attrs, "gen_ai.request.id", "unknown").rsplit("-0", 1)[0]
                    if request_id in requestIDs:
                        prompt_tokens = int(get_val(attrs, "gen_ai.usage.prompt_tokens", 0))
                        completion_tokens = int(get_val(attrs, "gen_ai.usage.completion_tokens", 0))
                        
                        # Phase latencies
                        prefill_latency = float(get_val(attrs, "gen_ai.latency.time_in_model_prefill", 0))
                        decode_latency = float(get_val(attrs, "gen_ai.latency.time_in_model_decode", 0))
                        if prefill_latency > 0 and decode_latency > 0:
                            # 1. Create Prefill Row
                            # Prefill starts at start_time + time_in_queue
                            # Prefill ends at start_time + prefill_latency
                            prefill_start = start_time + float(get_val(attrs, "gen_ai.latency.time_in_queue", 0))
                            prefill_end = start_time + prefill_latency
                            rows.append({
                                "request_id": request_id,
                                "phase_type": "prefill",
                                "t_start": (prefill_start - global_min_start_time)*1e6,
                                "t_end": (prefill_end - global_min_start_time)*1e6,
                                "prefill_tokens": prompt_tokens,
                                "decode_tokens": 0
                            })

                            # 2. Create Decode Row
                            # Decode starts where prefill ended
                            # Decode starts where prefill ended
                            decode_start = prefill_end
                            decode_end = prefill_end + decode_latency
                            rows.append({
                                "request_id": request_id,
                                "phase_type": "decode",
                                "t_start": (decode_start - global_min_start_time)*1e6,
                                "t_end": (decode_end - global_min_start_time)*1e6,
                                "prefill_tokens": 0,
                                "decode_tokens": completion_tokens
                            })

    # Write to CSV
    fieldnames = ["request_id", "phase_type", "t_start", "t_end", "prefill_tokens", "decode_tokens"]
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Successfully processed {len(rows)} phase entries into {output_file}")

if __name__ == "__main__":
    train_path = os.path.join("test", "*/")
    all_train = glob.glob(train_path)
    for train_path in all_train:
        traces_path = os.path.join(train_path, "traces.json")
        sweep_info_filepath = os.path.join(train_path, "sweep_info.json")
        # read GuideLLM sweep info and find relevant requestIDs
        requestIDs = []
        try:
            with open(sweep_info_filepath, 'r') as f:
                sweep_info = json.load(f)
        except:
            print("Could not read sweep info file.")
            sys.exit()
        for sweep in sweep_info:
            requestIDs.extend(sweep["requestIDs"])
        print(f"GuideLLM benchmark has a total of {len(requestIDs)} requests. Processing phases...")
        vllm_phases_filepath = os.path.join(train_path, "vllm_phases.csv")
        global_min_start_time = find_global_min_start(traces_path, requestIDs)
        process_traces(traces_path, vllm_phases_filepath, requestIDs, global_min_start_time)