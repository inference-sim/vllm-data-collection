"""
Script for training BLIS's alpha model in Scenario5 only on traces data
"""

import argparse
import json
import os
import pickle
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from postprocessing_utils import ALPHA_METRICS_FILENAME, ALPHA_WEIGHTS_FILENAME, BLIS_TRAINING_FILEPATH

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

def train_alpha_model(results_path, model_path):
    """
    Linear Regression model:
    alpha0 = mean(e2e_time - (queued + prefill + decode))
    """
    # get training data for alpha model
    training_data_filename = os.path.join(results_path, BLIS_TRAINING_FILEPATH)
    try:
        with open(training_data_filename, 'r') as f:
            training_data = json.load(f)
    except:
        print("Could not load BLIS training data.")
        sys.exit()
    processing_times = []
    input_lengths = []
    for benchmark in training_data["benchmarks"]:
        all_processing_times_microsec = [x*1e6 for x in benchmark["all_processing_times(s)"]]
        processing_times.extend(all_processing_times_microsec) # in microsecs for BLIS
        input_lengths.extend(benchmark["all_input_lens"])

    input_features = [[1, input_length] for input_length in input_lengths]
    alpha_model = LinearRegression(positive=True, fit_intercept=False)
    alpha_model.fit(input_features, processing_times)


    metrics_coeffs = get_metrics_and_coeffs(input_features, processing_times, alpha_model) # alpha0, alpha1
    print("Alpha training complete.")
    print(metrics_coeffs)

    # save alpha model weights
    alpha_model_weights_filename = os.path.join(model_path, ALPHA_WEIGHTS_FILENAME)
    with open(alpha_model_weights_filename, 'wb') as file:
        pickle.dump(alpha_model, file)
    print(f"Model saved to {alpha_model_weights_filename}")

    # save model metrics and coefficients
    alpha_model_metrics_filename = os.path.join(model_path, ALPHA_METRICS_FILENAME)
    with open(alpha_model_metrics_filename, 'w+') as file:
        json.dump(metrics_coeffs, file, indent=4)
    print(f"Alpha model metrics and coefficients saved to {alpha_model_weights_filename}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Read and parse traces JSON file.")
    parser.add_argument("--model_path", 
                        help="Folder path to save trained alpha model")
    parser.add_argument("--train_results_path",
                        default=".", 
                        help="Location to get training data from.")
    args = parser.parse_args()

    train_alpha_model(args.train_results_path, args.model_path)