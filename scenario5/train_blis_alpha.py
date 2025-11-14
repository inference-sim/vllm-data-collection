"""
Script for training BLIS's alpha model in Scenario5 only on traces data
"""

import argparse
import json
import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

WEIGHTS_FILENAME = 'BLIS_alpha_weights.pkl'
METRICS_FILENAME = 'BLIS_alpha_metrics.json'

def get_metrics_and_coeffs(X_train, y_train, alpha_model):
    """
    Save coefficients with R2-score, MAE and MAPE for alpha model
    """
    results = {"type": "BLIS_alpha_train"}
    training_score = alpha_model.score(X_train, y_train)
    training_preds = alpha_model.predict(X_train)
    training_mae = round(mean_absolute_error(training_preds, y_train), 3)
    training_mape = round(mean_absolute_percentage_error(training_preds, y_train), 3)

    results["train_r2"] = training_score
    results["train_mae"] = training_mae
    results["train_mape"] = training_mape
    results["coeffs"] = list(alpha_model.coef_)
    return results

def postprocess_delays(filepath):
    """
    Postprocess traces file to get server-side delay metrics,
    like e2e_latency, prefill_time, decode_time, queued_time.
    """
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
    """
    Linear Regression model:
    alpha0 + alpha1 * input_len = e2e_time - (queued + prefill + decode)
    """
    processing_times = []
    input_lengths = []
    for request in all_requests:
        processing_times.append(request["e2e_latency"] - (request["queued_time"] + request["prefill_time"] + request["decode_time"]))
        input_lengths.append([request["input_tokens"], 1])
    alpha_model = LinearRegression(positive=True, fit_intercept=False)
    alpha_model.fit(input_lengths, processing_times)

    metrics_coeffs = get_metrics_and_coeffs(input_lengths, processing_times, alpha_model)
    return alpha_model, metrics_coeffs

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Read and parse traces JSON file.")
    parser.add_argument("--traces", help="Path to the vllm traces file to be read.")
    parser.add_argument("--results_path",
                            default=".", 
                            help="Location to save alpha model")
    args = parser.parse_args()

    all_requests = postprocess_delays(args.traces)
    alpha_model, metrics_coeffs = train_alpha_model(all_requests)
    print("Alpha training completed.")
    print(metrics_coeffs)

    # save alpha model weights
    alpha_model_weights_filename = os.path.join(args.results_path, WEIGHTS_FILENAME)
    with open(alpha_model_weights_filename, 'wb') as file:
        pickle.dump(alpha_model, file)
    print(f"Model saved to {alpha_model_weights_filename}")

    # save model metrics and coefficients
    alpha_model_metrics_filename = os.path.join(args.results_path, METRICS_FILENAME)
    with open(alpha_model_metrics_filename, 'w+') as file:
        json.dump(metrics_coeffs, file, indent=4)
    print(f"Model metrics and coefficients saved to {alpha_model_weights_filename}")


