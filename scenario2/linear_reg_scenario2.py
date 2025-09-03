import argparse
import json
import os
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, RANSACRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLOCK_SIZE = 16

def process_results(data, chunk_size):
    processed_X = []
    processed_Y = []
    processed_chunk_sizes = []
    workload_data = data["workloads"]
    for workload in workload_data:
        for idx, pair in enumerate(workload["prompt_len_pairs"]):
            X_features = np.column_stack([workload["m"],
                                          pair[1] - pair[0], # delta
                                          workload["m"] * chunk_size, 
                                          (workload["m"] * workload["m"] * chunk_size * chunk_size)/(2*BLOCK_SIZE*BLOCK_SIZE),
                                          (workload["m"]*chunk_size * chunk_size)/(2*BLOCK_SIZE*BLOCK_SIZE),
                                          ((pair[1] - pair[0]) * workload["m"] * chunk_size)/(BLOCK_SIZE * BLOCK_SIZE),
                                          ((pair[1] - pair[0]) * (pair[1] - pair[0]))/(BLOCK_SIZE*BLOCK_SIZE)
                                          ])
            y = np.array([workload["e2e_pairs"][idx][0], workload["e2e_pairs"][idx][1]])
            processed_X.append(X_features)
            processed_Y.append(y)
            processed_chunk_sizes.append(chunk_size)
    return processed_X, processed_Y, processed_chunk_sizes

def remove_outliers(X_train, y_train):
    # filter out anomalies
    print ("Dataset size before filtering: ", X_train.shape[0])
    mean_y = np.mean(y_train)
    std_y = np.std(y_train)
    z_scores = (y_train - mean_y) / std_y
    filter_mask = np.abs(z_scores) < 2
    y_train = y_train[filter_mask]
    X_train = X_train[filter_mask]
    print ("Dataset size after filtering: ", X_train.shape[0])
    return X_train, y_train

def aggregate_results(model_name, scenario):
    chunk_sizes = [256, 512, 1024, 2048, 4096]
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    chunk_sizes_train = []
    chunk_sizes_test = []
    for chunk_size in chunk_sizes:
        local_folder_name = f'../results_new/{scenario}/{model_name}'
        items = os.listdir(local_folder_name)
        run_folders = [item for item in items if os.path.isdir(os.path.join(local_folder_name, item))]

        for run_folder in run_folders:
            with open(f'{local_folder_name}/{run_folder}/chunk_size_{chunk_size}/results/{scenario}_output_train.json', 'r') as f:
                train_data_json = json.load(f)
                X_train_curr, y_train_curr, chunk_sizes_train_curr = process_results(train_data_json, chunk_size)
                X_train.extend(X_train_curr)
                y_train.extend(y_train_curr)
                chunk_sizes_train.extend(chunk_sizes_train_curr)

            with open(f'{local_folder_name}/{run_folder}/chunk_size_{chunk_size}/results/{scenario}_output_test.json', 'r') as file:
                test_data_json = json.load(file)
                X_test_curr, y_test_curr, chunk_sizes_test_curr = process_results(test_data_json, chunk_size)
                X_test.extend(X_test_curr)
                y_test.extend(y_test_curr)
                chunk_sizes_test.extend(chunk_sizes_test_curr)
    return X_train, y_train, X_test, y_test, chunk_sizes_train, chunk_sizes_test

def plot_model_results(LLM, plots_path, X_train, y_train, X_test, y_test, chunk_sizes_train, chunk_size_test, model, model_type: str):
    training_score = round(model.score(X_train, y_train), 3)
    test_score = round(model.score(X_test, y_test), 3)
    training_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    training_mae = round(mean_absolute_error(training_preds, y_train), 3)
    training_mape = round(mean_absolute_percentage_error(training_preds, y_train), 3)
    test_mae = round(mean_absolute_error(test_preds, y_test), 3)
    test_mape = round(mean_absolute_percentage_error(test_preds, y_test), 3)

    print (f"##################### {model_name} ############################")
    print(f"{model_type} Model Train Score: {training_score}")
    print(f"{model_type} Model Train MAE: {training_mae}")
    print(f"{model_type} Model Train MAPE: {training_mape}")

    print(f"{model_type} Model Test Score: {test_score}")
    print(f"{model_type} Model Test MAE: {test_mae}")
    print(f"{model_type} Model Test MAPE: {test_mape}")

    fig, axs = plt.subplots(3, 2, figsize=(20, 30))
    row_order = ['C', 'm', 'delta']

    for idx, row in enumerate(row_order):
        if idx == 0:
            train_data = np.array(chunk_sizes_train)
            test_data = np.array(chunk_size_test)
        else:
            train_data = X_train[:, idx - 1]
            test_data = X_test[:, idx - 1]
        axs[idx, 0].scatter(train_data, y_train[:, 0])
        axs[idx, 0].scatter(train_data, y_train[:, 1])
        axs[idx, 0].set_title(f'Train R2 {model_type}: {training_score}, \nTrain MAE: {training_mae}, \nTrain MAPE: {training_mape}')
        axs[idx, 0].set_xlabel(row)
        axs[idx, 0].set_ylabel('e2e latency')
        axs[idx, 0].legend(["Req1", "Req2"])
        axs[idx, 0].grid(True)

        axs[idx, 1].scatter(test_data, y_test[:, 0])
        axs[idx, 1].scatter(test_data, y_test[:, 1])
        axs[idx, 1].set_title(f'Test R2 {model_type}: {test_score}, \nTest MAE: {test_mae}, \nTest MAPE: {test_mape}')
        axs[idx, 1].set_xlabel(row)
        axs[idx, 1].set_ylabel('e2e latency')
        axs[idx, 1].legend(["Req1", "Req2"])
        axs[idx, 1].grid(True)

    fig.suptitle(f'Results for {LLM}')

    plt.savefig(f"{plots_path}/{LLM}_{model_type}.png")

def train_lr(model_name, scenario, plots_path):
    
    X_train, y_train, X_test, y_test, chunk_sizes_train, chunk_sizes_test = aggregate_results(model_name, scenario)
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
    X_train = np.squeeze(X_train)
    X_test = np.squeeze(X_test)
    
    model_lr = LinearRegression(positive=True)
    model_lr.fit(X_train, y_train)

    # ransac = RANSACRegressor(random_state=0, estimator=model_lr).fit(X_train, y_train)

    plot_model_results(model_name, plots_path, X_train, y_train, X_test, y_test, chunk_sizes_train, chunk_sizes_test, model_lr, "LR")
    # plot_model_results(model_name, plots_path, X_train, y_train, X_test, y_test, chunk_sizes_train, chunk_sizes_test, ransac, "RANSAC")

    # print (f"LR coefficients: {model_lr.coef_}")
    # print (f"LR intercept: {model_lr.intercept_}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple vLLM Benchmark Runner')
    parser.add_argument('--scenario', help='scenario X',  default="scenario2")
    args = parser.parse_args()
    models = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-Instruct-v0.1", "google/gemma-7b", "meta-llama/Llama-3.1-8B","ibm-granite/granite-3.3-8b-instruct", "Qwen/Qwen3-14B", "mistralai/Mistral-Small-24B-Instruct-2501", "Qwen/Qwen3-32B"]
    for model in models:
        model_name = model.split("/")[-1].replace(".", "_")
        plots_path = f"../plots_joint/{args.scenario}/{model_name}"
        os.makedirs(plots_path, exist_ok=True)
        train_lr(model_name, args.scenario, plots_path)



