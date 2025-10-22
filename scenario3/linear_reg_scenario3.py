import argparse
import json
import os
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, RANSACRegressor
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

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
        processed_chunk_sizes.extend([chunk_size, chunk_size, chunk_size, chunk_size])
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
            try:
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
            except:
                print (f"Data not found for {model_name}, chunk_size = {chunk_size}. Skipping...")
    return X_train, y_train, X_test, y_test, chunk_sizes_train, chunk_sizes_test

def calculate_metrics(X_train, y_train, X_test, y_test, model, mode = "overall"):
    """
    Get R2-score, MAE and MAPE
    """
    training_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    training_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    training_mae = round(mean_absolute_error(training_preds, y_train), 3)
    training_mape = round(mean_absolute_percentage_error(training_preds, y_train), 3)
    test_mae = round(mean_absolute_error(test_preds, y_test), 3)
    test_mape = round(mean_absolute_percentage_error(test_preds, y_test), 3)
    if mode == "overall":
        caption = f"##################### {model_name}-overall ############################"
    else:
        caption = f"##################### {model_name}-{mode} ############################"
    print (caption)
    print(f"LR Model Train Score: {training_score}")
    print(f"LR Model Train MAE: {training_mae}")
    print(f"LR Model Train MAPE: {training_mape}")

    print(f"LR Model Test Score: {test_score}")
    print(f"LR Model Test MAE: {test_mae}")
    print(f"LR Model Test MAPE: {test_mape}")

    print("Coefficients: ", model.coef_)
    feature_imp = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    print("Feature Importances (Mean):")
    for i, importance in enumerate(feature_imp.importances_mean):
        print(f"Feature {i}: {importance:.4f}")
    print("\nFeature Importances (Standard Deviation):")
    for i, std_dev in enumerate(feature_imp.importances_std):
        print(f"Feature {i}: {std_dev:.4f}")

def plot_model_results(LLM, plots_path, X_train, y_train, X_test, y_test, model, model_type: str, chunk_sizes_train, chunk_sizes_test):
    """
    Print LR scores (R2, MAE, MAPE) and plot results
    """

    for idx in range(4):
        X_train_req_i = X_train[idx::4]
        y_train_req_i = y_train[idx::4]
        X_test_req_i = X_test[idx::4]
        y_test_req_i = y_test[idx::4]
        color_maps = {1024: 'red', 2048: 'blue', 4096: 'green'}
        chunk_sizes_train_req_i = chunk_sizes_train[idx::4]
        chunk_sizes_test_req_i = chunk_sizes_test[idx::4]
        colors_train = [color_maps[i] for i in chunk_sizes_train_req_i]
        colors_test = [color_maps[i] for i in chunk_sizes_test_req_i]

        # print (max(abs(training_preds - y_train)))
        # print (max(abs(test_preds - y_test)))
        # calculate_metrics(X_train_req_i, y_train_req_i, X_test_req_i, y_test_req_i, model, f"request-{idx+1}")

        training_preds_req_i = model.predict(X_train_req_i)
        residuals_training_req_i = np.square(training_preds_req_i - y_train_req_i)

        if idx == 0:
            fig, axs = plt.subplots(1, 2, figsize=(20, 10))
            input_lens_req1 = X_train_req_i[:,0]
            axs[0].scatter(input_lens_req1, y_train_req_i, color=colors_train)
            axs[1].scatter(input_lens_req1%128, residuals_training_req_i, color=colors_test)
            axs[0].set_xlabel("Input len")
            axs[0].set_title(f'e2e latency vs input len')
            axs[1].set_xlabel("Input len % 128")
            axs[1].set_title(f'Prediction Residual vs input len')

            legend_elements = [Patch(facecolor=color, edgecolor=color, label=f'chunk_size={size}') 
                       for size, color in color_maps.items()]

            axs[0].set_ylabel('e2e latency')
            axs[0].legend(handles=legend_elements)
            axs[0].grid(True)
            
            axs[1].set_ylabel('Prediction Residual')
            axs[1].legend(handles=legend_elements)
            axs[1].grid(True)

            fig.suptitle(f'Results for {LLM}')

            plt.savefig(f"{plots_path}/{LLM}_{model_type}_request1.png")
            plt.close()
        
        elif idx == 1:
            fig, axs = plt.subplots(3, 2, figsize=(20, 20))
            for row in range(3):
                if row == 0:
                    X_vals = X_train_req_i[:,0]
                    x_label = "Total Input len"
                if row == 1:
                    X_vals = X_train_req_i[:,3]
                    x_label = "Cached Input Tokens"
                if row == 2:
                    X_vals = X_train_req_i[:,4]
                    x_label = "Uncached Input Tokens"
                axs[row,0].scatter(X_vals, y_train_req_i, color=colors_train)
                axs[row,1].scatter(X_vals, residuals_training_req_i, color=colors_test)
                axs[row,0].set_xlabel(x_label)
                axs[row,0].set_title(f'e2e latency vs {x_label}')
                axs[row,1].set_xlabel(x_label)
                axs[row,1].set_title(f'Prediction Residual vs {x_label}')

                legend_elements = [Patch(facecolor=color, edgecolor=color, label=f'chunk_size={size}') 
                        for size, color in color_maps.items()]

                axs[row,0].set_ylabel('e2e latency')
                axs[row,0].legend(handles=legend_elements)
                axs[row,0].grid(True)
                
                axs[row,1].set_ylabel('Prediction Residual')
                axs[row,1].legend(handles=legend_elements)
                axs[row,1].grid(True)


            fig.suptitle(f'Results for {LLM}')

            plt.savefig(f"{plots_path}/{LLM}_{model_type}_request2.png")
            plt.close()



    calculate_metrics(X_train, y_train, X_test, y_test, model)

def train_lr_by_request_groups(model_name, scenario, plots_path):
    X_train, y_train, X_test, y_test, chunk_sizes_train, chunk_sizes_test = aggregate_results_joint(model_name, scenario)
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
    
    for idx in range(4):
        X_train_req_i = X_train[idx::4]
        y_train_req_i = y_train[idx::4]
        X_test_req_i = X_test[idx::4]
        y_test_req_i = y_test[idx::4]
        
        model_lr_req_i = LinearRegression(positive=True)
        model_lr_req_i.fit(X_train_req_i, y_train_req_i)
        calculate_metrics(X_train_req_i, y_train_req_i, X_test_req_i, y_test_req_i, model_lr_req_i, f"request-{idx + 1}")
        # prev_error = float("Inf")
        # curr_error = mean_absolute_percentage_error(model_lr.predict(X_train), y_train)
        # max_iters = 0
        # it = 1
        # while curr_error < prev_error and it < max_iters:
        #     prev_error = curr_error
        #     training_preds = model_lr.predict(X_train)
        #     sample_weights = 1/(np.maximum(np.square(training_preds - y_train), 1e-6))
        #     model_lr.fit(X_train, y_train, sample_weight=sample_weights)
        #     curr_error = mean_absolute_percentage_error(model_lr.predict(X_train), y_train)
        #     it += 1

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
    df.to_csv(f"results_processed/train_{model_name}_scenario3_full.csv", index = False)
    model_lr = LinearRegression(positive=True)
    model_lr.fit(X_train, y_train)
    prev_error = float("Inf")
    curr_error = mean_absolute_percentage_error(model_lr.predict(X_train), y_train)
    max_iters = 30
    it = 1
    while curr_error < prev_error and it < max_iters:
        prev_error = curr_error
        training_preds = model_lr.predict(X_train)
        sample_weights = 1/(np.maximum(np.square(training_preds - y_train), 1e-6))
        model_lr.fit(X_train, y_train, sample_weight=sample_weights)
        curr_error = mean_absolute_percentage_error(model_lr.predict(X_train), y_train)
        it += 1

    # ransac = RANSACRegressor(random_state=0, estimator=model_lr).fit(X_train, y_train)

    plot_model_results(model_name, plots_path, X_train, y_train, X_test, y_test, model_lr, "LR", chunk_sizes_train, chunk_sizes_test)

    # print (f"LR coefficients: {model_lr.coef_}")
    # print (f"LR intercept: {model_lr.intercept_}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple vLLM Benchmark Runner')
    parser.add_argument('--scenario', help='scenario X',  default="scenario3")
    args = parser.parse_args()
    models = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B", "google/gemma-7b", "mistralai/Mistral-7B-Instruct-v0.1", "meta-llama/Llama-3.1-8B", "ibm-granite/granite-3.3-8b-instruct", "Qwen/Qwen3-14B", "mistralai/Mistral-Small-24B-Instruct-2501", "Qwen/Qwen3-32B"]
    for model in models:
        model_name = model.split("/")[-1].replace(".", "_")
        plots_path = f"../plots_vstack_new/{args.scenario}/{model_name}"
        os.makedirs(plots_path, exist_ok=True)
        train_lr(model_name, args.scenario, plots_path)
