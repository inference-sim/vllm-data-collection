# use this file only in train/val mode

import os
import json

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

from experiment_configs_constants_train import *

def calculate_metrics(X_train, y_train, X_test, y_test, busy_loop_model, model_name, mode, include_val = False):
    """
    Get R2-score, MAE and MAPE
    """
    training_score = busy_loop_model.score(X_train, y_train)
    training_preds = busy_loop_model.predict(X_train)
    training_mae = round(mean_absolute_error(training_preds, y_train), 3)
    training_mape = round(mean_absolute_percentage_error(training_preds, y_train), 3)

    caption = f"##################### {model_name}-alphas-{mode} ############################"
    print(caption)
    print(f"LR Model Train Score: {training_score}")
    print(f"LR Model Train MAE: {training_mae}")
    print(f"LR Model Train MAPE: {training_mape}")
    print(f"Coeffs: {busy_loop_model.coef_}")

    if include_val:
        test_score = busy_loop_model.score(X_test, y_test)
        test_preds = busy_loop_model.predict(X_test)
        test_mae = round(mean_absolute_error(test_preds, y_test), 3)
        test_mape = round(mean_absolute_percentage_error(test_preds, y_test), 3)

        print(f"LR Model Test Score: {test_score}")
        print(f"LR Model Test MAE: {test_mae}")
        print(f"LR Model Test MAPE: {test_mape}")

def plot_delays(model_name, mode, rr):
    for spec in SPECS:
        queuing_delays = []
        finished_delays = []
        input_lens = []
        output_lens = []
        for mbnt in MAX_NUM_BATCHED_TOKENS:
            spec_small = spec.lower()
            rr = f"{float(rr):.2f}"
            results_folder = f"../results_new/scenario4/{model_name}/{mode}/{spec_small}/mbnt_{mbnt}/rr_{rr}"
            if os.path.isdir(results_folder):
                for dirpath, _, filenames in os.walk(results_folder):
                    for filename in filenames:
                        if filename == f"detailed_results_{mode}.json":
                            full_path = os.path.join(dirpath, filename)
                            with open (full_path, "r") as f:
                                json_data = json.load(f)
                            prompts = json_data["prompts"]
                            for prompt in prompts:
                                if prompt["error"]=="": #check for errors
                                    for event in prompt["events"]:
                                        if event["event_type"] == "FINISHED":
                                            finished_time = event["timestamp"]
                                        if event["event_type"] == "SERVER_HIT":
                                            server_hit_time = event["timestamp"]
                                        if event["event_type"] == "QUEUED":
                                            queue_time = event["timestamp"]
                                        if event["event_type"] == "SERVER_LEFT":
                                            server_left_time = event["timestamp"]
                                    server_hit_to_queue_delay = (queue_time - server_hit_time)*1e6
                                    finished_to_server_left_delay = (server_left_time - finished_time)*1e6
                                    queuing_delays.append(server_hit_to_queue_delay)
                                    finished_delays.append(finished_to_server_left_delay)
                                    input_lens.append(prompt["input_len"])
                                    output_lens.append(prompt["output_len"])
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(1, 2, figsize=(30, 20))
        label = f'Spec={spec} delays'
        queuing_coeffs = {
            "Qwen2_5-7B": [2.66696657e+00, 5.52881440e+03], # change to alpha coefficients you want to plot
            "Qwen3-14B": [2.40788681e+00, 9.08991938e+03]
        }

        finished_coeffs = {
            "Qwen2_5-7B": [2.88576689e-01, 6.25360201e+02], # change to alpha coefficients you want to plot
            "Qwen3-14B": [1.03981306, 796.73460409]
        }
        
        queuing_coeff = queuing_coeffs[model_name]
        finished_coeff = finished_coeffs[model_name]
        ax[0].scatter(input_lens, queuing_delays, marker='o', linestyle='-', label=label)
        ax[0].set_title(f'Queuing delay for spec={spec}, model={model_name}', fontsize=16)
        ax[0].set_xlabel('Input len', fontsize=12)
        ax[0].set_ylabel('Server hit to queue delay', fontsize=12)
        predicted_queuing_delays = queuing_coeff[0] * np.array(input_lens) + queuing_coeff[1]

        ax[0].plot(input_lens, predicted_queuing_delays, color='red', marker='o')
        ax[1].scatter(output_lens, finished_delays, marker='o', linestyle='-', label=label)
        ax[1].set_title(f'Finished delay for spec={spec}, model={model_name}', fontsize=16)
        ax[1].set_xlabel('Output len', fontsize=12)
        ax[1].set_ylabel('Finished to server left delay', fontsize=12)
        predicted_finished_delays = finished_coeff[0] * np.array(output_lens) + finished_coeff[1]
        ax[1].plot(output_lens, predicted_finished_delays, color='red', marker='o')
        os.makedirs("delay_plots", exist_ok=True)
        plt.savefig(f"delay_plots/{model_name}_spec={spec}_alpha_delays.png")

        plt.close()

def get_delays(model_name, mode):
    queuing_delays = []
    finished_delays = []
    input_lens = []
    output_lens = []
    model_level_queuing_delay = 0
    model_level_queuing_num = 0
    model_level_finished_delay = 0
    model_level_finished_num = 0
    for spec in SPECS:
        model_spec_level_queuing_delay = 0
        model_spec_level_queuing_num = 0
        model_spec_level_finished_delay = 0
        model_spec_level_finished_num = 0
        for rr in REQUEST_RATES[spec]:
            for mbnt in MAX_NUM_BATCHED_TOKENS:
                spec_small = spec.lower()
                rr = f"{float(rr):.2f}"
                results_folder = f"../results_new/scenario4/{model_name}/{mode}/{spec_small}/mbnt_{mbnt}/rr_{rr}"
                if os.path.isdir(results_folder):
                    for dirpath, _, filenames in os.walk(results_folder):
                        for filename in filenames:
                            if filename == f"detailed_results_{mode}.json":
                                full_path = os.path.join(dirpath, filename)
                                with open (full_path, "r") as f:
                                    json_data = json.load(f)
                                prompts = json_data["prompts"]
                                for prompt in prompts:
                                    if prompt["error"]=="": #check for errors
                                        for event in prompt["events"]:
                                            if event["event_type"] == "FINISHED":
                                                finished_time = event["timestamp"]
                                            if event["event_type"] == "SERVER_HIT":
                                                server_hit_time = event["timestamp"]
                                            if event["event_type"] == "QUEUED":
                                                queue_time = event["timestamp"]
                                            if event["event_type"] == "SERVER_LEFT":
                                                server_left_time = event["timestamp"]
                                        server_hit_to_queue_delay = queue_time - server_hit_time
                                        finished_to_server_left_delay = server_left_time - finished_time
                                        model_spec_level_queuing_delay += server_hit_to_queue_delay * 1e6
                                        model_level_queuing_delay += server_hit_to_queue_delay * 1e6
                                        queuing_delays.append(server_hit_to_queue_delay * 1e6)
                                        finished_delays.append(finished_to_server_left_delay * 1e6)
                                        input_lens.append([prompt["input_len"], 1])
                                        output_lens.append([prompt["output_len"], 1])
                                        model_spec_level_queuing_num += 1
                                        model_spec_level_finished_num += 1
                                        model_spec_level_finished_delay += finished_to_server_left_delay * 1e6
                                        model_level_finished_delay += finished_to_server_left_delay * 1e6
                                        model_level_queuing_num += 1
                                        model_level_finished_num += 1
        print(f"#####################{model_name}_{spec}#########################")
        print(f"Avg. Server hit to queue delay: {model_spec_level_queuing_delay/model_spec_level_queuing_num}")
        print(f"Avg. finished to Server left delay: {model_spec_level_finished_delay/model_spec_level_finished_num}")
    print(f"#####################{model_name}_overall#########################")
    print(f"Avg. Server hit to queue delay: {model_level_queuing_delay/model_level_queuing_num}")
    print(f"Avg. finished to Server left delay: {model_level_finished_delay/model_level_finished_num}")
    return queuing_delays, finished_delays, input_lens, output_lens

if __name__=="__main__":
    model_name = MODEL.split("/")[-1].replace(".", "_")

    # saturation doesn't matter for EDA
    y_queue_train, y_finished_train, X_queue_train, X_finished_train = get_delays(model_name, "train")
    # y_queue_test, y_finished_test, X_queue_test, X_finished_test = get_delays(model_name, "val")

    model_queue = LinearRegression(positive=True, fit_intercept=False)
    model_queue.fit(X_queue_train, y_queue_train)
    model_finished = LinearRegression(positive=True, fit_intercept=False)
    model_finished.fit(X_finished_train, y_finished_train)

    calculate_metrics(X_queue_train, y_queue_train, [], [], model_queue, model_name, "queueing_delay")
    calculate_metrics(X_finished_train, y_finished_train, [], [], model_finished, model_name, "finished_delay")