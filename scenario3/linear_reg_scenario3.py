import argparse
import json
import os
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, RANSACRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_configs_constants import *

def process_results_joint(data, chunk_size):
    """
    Processes json results into necessary training format with features as per model
    Args:
         data: JSON file
         chunk_size: int
         mode: hstack for stacking entries of a request pair into columns
               vstack for stacking entries of a request pair into rows
    """
    processed_X = []
    processed_Y = []
    processed_chunk_sizes = []
    workload_data = data["workloads"]
    for workload in workload_data:
        delta1, delta2, delta3 = workload["deltas"]
        _, _, l_out_3, l_out_4 = workload["output_len_quads"]
        q1 = (delta1 + 4)//chunk_size
        r1 = (delta1 + 4 + chunk_size - 1)//chunk_size
        s1 = (delta1 + 4)/chunk_size - q1
        beta3_term_1 = chunk_size * chunk_size * ((q1 * (q1 + 1))/2 + q1 * s1 + s1 * s1)
        req1_features = [delta1 + 4, 1, r1, 0, (delta1 + 4), beta3_term_1, 0, 0, 1, delta1 + 5]
        
        q2 = (delta2 + 1)//chunk_size
        r2 = (delta2 + 1 + chunk_size - 1)//chunk_size
        s2 = (delta2 + 1)/chunk_size - q2
        beta3_term_2 = (delta1 + 4) * (delta2 + 1) + chunk_size * chunk_size * ((q2 * (q2 + 1))/2 + q2 * s2 + s2 * s2)
        req2_features = [5 + delta1 + delta2, 1, r2, delta1 + 4, delta2 + 1, beta3_term_2, 0, 0, 1, delta1 + delta2 + 6]

        beta5_term_3 = (l_out_3 * (8 + 2 * delta1 + l_out_3 - 1))/2
        req3_features = [delta1 + 4, l_out_3, l_out_3, delta1 + 4, 0, 0, l_out_3, beta5_term_3, 1, delta1 + l_out_3 + 4]

        q4 = (delta2 + 1)//chunk_size
        r4 = (delta2 + 1 + chunk_size - 1)//chunk_size
        s4 = (delta2 + 1)/chunk_size - q4
        beta3_term_4 = (delta1 + 4) * (delta2 + 1) + chunk_size * chunk_size * ((q4 * (q4 + 1))/2 + q4 * s4 + s4 * s4)
        beta5_term_4 = ((l_out_4 - 1) * (10 + 2 * delta1 + 2 * delta2 + l_out_4 - 2))/2
        req4_features = [delta1 + delta2 + 5, l_out_4, r4 + l_out_4 - 1, delta1 + 4, delta2 + 1, beta3_term_4, l_out_4 - 1, beta5_term_4, 1, delta1 + delta2 + l_out_4 + 5]

        X_features = [req1_features, req2_features, req3_features, req4_features]
        y = workload["e2e_quads"]
        processed_chunk_sizes.extend([chunk_size, chunk_size])
        processed_chunk_sizes.append(chunk_size)
        processed_X.extend(X_features)
        processed_Y.extend(y)
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

def aggregate_results_joint(model_name, scenario):
    """
    Aggregates results across multiple runs of the same LLM into X_train, y_train, X_test, y_test
    Args:
         model_name: str
         scenario: str
         mode: hstack for stacking entries of a request pair into columns
               vstack for stacking entries of a request pair into rows
    """
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    chunk_sizes_train = []
    chunk_sizes_test = []
    for chunk_size in CHUNK_SIZES:
        local_folder_name = f'../results_new/{scenario}/{model_name}'
        items = os.listdir(local_folder_name)
        run_folders = [item for item in items if os.path.isdir(os.path.join(local_folder_name, item))]

        for run_folder in run_folders:
            with open(f'{local_folder_name}/{run_folder}/chunk_size_{chunk_size}/results/{scenario}_output_train.json', 'r') as f:
                train_data_json = json.load(f)
                X_train_curr, y_train_curr, chunk_sizes_train_curr = process_results_joint(train_data_json, chunk_size)
                X_train.extend(X_train_curr)
                y_train.extend(y_train_curr)
                chunk_sizes_train.extend(chunk_sizes_train_curr)

            with open(f'{local_folder_name}/{run_folder}/chunk_size_{chunk_size}/results/{scenario}_output_test.json', 'r') as file:
                test_data_json = json.load(file)
                X_test_curr, y_test_curr, chunk_sizes_test_curr = process_results_joint(test_data_json, chunk_size)
                X_test.extend(X_test_curr)
                y_test.extend(y_test_curr)
                chunk_sizes_test.extend(chunk_sizes_test_curr)
    return X_train, y_train, X_test, y_test, chunk_sizes_train, chunk_sizes_test

def plot_model_results(LLM, plots_path, X_train, y_train, X_test, y_test, model, model_type: str):
    """
    Print LR scores (R2, MAE, MAPE) and plot results
    """
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

    # fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    
    # axs[0].scatter(y_train)
    # axs[0].plot(training_preds)
    # axs[1].plot(y_test)
    # axs[1].plot(test_preds)

    # axs[0].set_title(f'Train R2 {model_type}: {training_score}, \nTrain MAE: {training_mae}, \nTrain MAPE: {training_mape}')
    # axs[0].set_xlabel("req index")
    # axs[0].set_ylabel('e2e latency')
    # axs[0].legend(["Orig", "Pred"])
    # axs[0].grid(True)
    
    # axs[1].set_title(f'Test R2 {model_type}: {test_score}, \nTest MAE: {test_mae}, \nTest MAPE: {test_mape}')
    # axs[1].set_xlabel("req index")
    # axs[1].set_ylabel('e2e latency')
    # axs[1].legend(["Orig", "Pred"])
    # axs[1].grid(True)

    # fig.suptitle(f'Results for {LLM}')

    # plt.savefig(f"{plots_path}/{LLM}_{model_type}.png")

def train_lr(model_name, scenario, plots_path):
    """
    Train linear regression model from aggregated results
    """
    X_train, y_train, X_test, y_test, chunk_sizes_train, chunk_sizes_test = aggregate_results_joint(model_name, scenario)
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
    vstack_cols = ["alpha_1", "alpha_2", "beta_0", "beta_1", "beta_2", "beta_3", "beta_4", "beta_5", "beta_6", "beta_7", "y"]
    data = np.column_stack([X_train, y_train])
    df = pd.DataFrame(data = data, columns=vstack_cols)
    # save X_train + y_train into a csv for validation
    df.to_csv(f"results_processed/train_{model_name}_scenario3.csv", index = False)
    model_lr = LinearRegression(positive=True)
    model_lr.fit(X_train, y_train)

    # ransac = RANSACRegressor(random_state=0, estimator=model_lr).fit(X_train, y_train)

    plot_model_results(model_name, plots_path, X_train, y_train, X_test, y_test, model_lr, "LR")

    # print (f"LR coefficients: {model_lr.coef_}")
    # print (f"LR intercept: {model_lr.intercept_}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple vLLM Benchmark Runner')
    parser.add_argument('--scenario', help='scenario X',  default="scenario3")
    args = parser.parse_args()
    models = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-Instruct-v0.1"]
    for model in models:
        model_name = model.split("/")[-1].replace(".", "_")
        plots_path = f"../plots_vstack/{args.scenario}/{model_name}"
        os.makedirs(plots_path, exist_ok=True)
        train_lr(model_name, args.scenario, plots_path)



