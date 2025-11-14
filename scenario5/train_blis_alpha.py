"""
Script for training BLIS's alpha model in Scenario5 only on traces data
"""

import argparse
import json
import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from postprocessing_utils import ALPHA_METRICS_FILENAME, ALPHA_WEIGHTS_FILENAME
from postprocessing_utils import read_traces_jsonl, get_server_side_metrics_from_traces

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

    traces_raw_data = read_traces_jsonl(args.traces)
    all_requests = get_server_side_metrics_from_traces(traces_raw_data)
    alpha_model, metrics_coeffs = train_alpha_model(all_requests)
    print("Alpha training complete.")
    print(metrics_coeffs)

    # save alpha model weights
    alpha_model_weights_filename = os.path.join(args.results_path, ALPHA_WEIGHTS_FILENAME)
    with open(alpha_model_weights_filename, 'wb') as file:
        pickle.dump(alpha_model, file)
    print(f"Model saved to {alpha_model_weights_filename}")

    # save model metrics and coefficients
    alpha_model_metrics_filename = os.path.join(args.results_path, ALPHA_METRICS_FILENAME)
    with open(alpha_model_metrics_filename, 'w+') as file:
        json.dump(metrics_coeffs, file, indent=4)
    print(f"Alpha model metrics and coefficients saved to {alpha_model_weights_filename}")


