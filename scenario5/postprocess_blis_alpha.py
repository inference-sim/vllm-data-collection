import json
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def calculate_metrics(X_train, y_train, alpha_model):
    """
    Get R2-score, MAE and MAPE for alpha model
    """
    training_score = alpha_model.score(X_train, y_train)
    training_preds = alpha_model.predict(X_train)
    training_mae = round(mean_absolute_error(training_preds, y_train), 3)
    training_mape = round(mean_absolute_percentage_error(training_preds, y_train), 3)

    caption = f"##################### alpha-model ############################"
    print(caption)
    print(f"LR Model Train Score: {training_score}")
    print(f"LR Model Train MAE: {training_mae}")
    print(f"LR Model Train MAPE: {training_mape}")
    print(f"Coeffs: {alpha_model.coef_}")

def get_delays(filepath):
    all_requests = []
    with open(filepath, 'r') as file:
        traces_raw_data = json.load(file)
        for resourceSpan in traces_raw_data["resourceSpans"]:
            request = {}
            for scopeSpan in resourceSpan["scopeSpans"]:
                for span in scopeSpan["spans"]:
                    for attribute in span["attributes"]:
                        if attribute["key"] == "gen_ai.request.id":
                            request["request_id"] = attribute["value"]
                        if attribute["key"] == "gen_ai.usage.prompt_tokens":
                            request["input_tokens"] = attribute["value"]
                        if attribute["key"] == "gen_ai.usage.completion_tokens":
                            request["output_tokens"] = attribute["value"]
                        if attribute["key"] == "gen_ai.latency.e2e":
                            request["e2e_latency"] = attribute["value"]
                        if attribute["key"] == "gen_ai.latency.time_in_queue":
                            request["queued_time"] = attribute["value"]
                        if attribute["key"] == "gen_ai.latency.time_in_model_prefill":
                            request["prefill_time"] = attribute["value"]
                        if attribute["key"] == "gen_ai.latency.time_in_model_decode":
                            request["decode_time"] = attribute["value"]
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
parser.add_argument("traces_filepath", help="Path to the vllm traces file to be read.")
args = parser.parse_args()

all_requests = get_delays(args.traces_filepath)
train_alpha_model(all_requests)


