import json
import os
from experiment_configs_constants import *
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

def get_model_mode_latencies(model_name, mode):
    model_mode_data = []
    for spec in SPECS:
        for chunk_size in CHUNK_SIZES:
            for rr in REQUEST_RATES:
                spec_small = spec.lower()
                results_folder = f"../results_new/scenario4/{model_name}/{spec_small}/chunk_size_{chunk_size}/rr_{rr}"
                if os.path.isdir(results_folder):
                    for dirpath, _, filenames in os.walk(results_folder):
                        for filename in filenames:
                            if filename == f"detailed_results_{mode}.json":
                                full_path = os.path.join(dirpath, filename)
                                with open (full_path, "r") as f:
                                    json_data = json.load(f)
                                prompts = json_data["prompts"]
                                for prompt in prompts:
                                    prompt_data = {"input_len": prompt["input_len"], "output_len": prompt["output_len"], 
                                                   "e2e_latency": prompt["e2e_latency"]}
                                    if prompt["error"]=="": #check for errors
                                        for event in prompt["events"]:
                                            if event["event_type"] == "FINISHED":
                                                finished_time = event["timestamp"]
                                                prompt_data["finished_step"] = event["step"]
                                            if event["event_type"] == "QUEUED":
                                                queued_time = event["timestamp"]
                                                prompt_data["queued_step"] = event["step"]
                                            if event["event_type"] == "SCHEDULED":
                                                scheduled_time = event["timestamp"]
                                                prompt_data["scheduled_step"] = event["step"]
                                        prompt_data["busy_loop_time_with_req"] = finished_time - scheduled_time
                                        prompt_data["busy_loop_time_waiting"] = scheduled_time - queued_time
                                        prompt_data["exp_path"] = dirpath
                                        model_mode_data.append(prompt_data) #only append if no error
    return model_mode_data

def get_step_compositions(mode, exp_path, start_step, end_step):
    step_data = {"beta_1": 0, "beta_2": 0, "beta_3": 0, "beta_4": 0, "beta_5": 0, "beta_6": 0, "beta_7": 0}
    start_file_idx = 1000*((start_step + 1000 - 1)//1000)
    end_file_idx = 1000*((end_step + 1000 - 1)//1000)
    for idx in range(start_file_idx, end_file_idx + 1, 1000):
        file_path = f"{exp_path}/metrics_{mode}_{idx}.json"
        try:
            with open(file_path, "r") as f:
                file_steps = json.load(f)
                for step in file_steps:
                    if int(step) >= start_step and int(step) <= end_step:
                        num_cache_miss_tokens = file_steps[step]["num_cache_miss_tokens"]
                        num_cache_hit_tokens = file_steps[step]["num_cache_hit_tokens"]
                        step_data["beta_1"] += num_cache_hit_tokens
                        step_data["beta_2"] += num_cache_miss_tokens
                        step_data["beta_3"] += num_cache_miss_tokens * num_cache_hit_tokens + num_cache_miss_tokens * num_cache_miss_tokens
                        step_data["beta_4"] += file_steps[step]["num_decode_reqs"]
                        step_data["beta_5"] += file_steps[step]["num_decode_reqs"]
                        step_data["beta_6"] += file_steps[step]["num_finished_reqs"]
                        step_data["beta_7"] += file_steps[step]["num_finished_tokens"]
        except:
            # print("Step data missing for request. Excluding this request...")
            pass
    return step_data

def processed_data(model_name, mode):
    model_mode_latencies = get_model_mode_latencies(model_name, mode)
    request_df = pd.DataFrame(model_mode_latencies)
    step_df = request_df[["exp_path", "scheduled_step", "finished_step", "busy_loop_time_with_req"]]
    step_df["steps"] = request_df.apply(lambda x: get_step_compositions(mode, x["exp_path"], x["scheduled_step"], x["finished_step"]), axis = 1)
    expanded_cols = step_df['steps'].apply(pd.Series)
    step_df = step_df.join(expanded_cols)
    step_df = step_df.drop(columns=["exp_path", "scheduled_step", "finished_step", "steps"])
    print(step_df.head())
    step_df.to_csv(f"train_df_{model_name}_{mode}.csv", index=False)
    return step_df

def calculate_metrics(X_train, y_train, X_test, y_test, busy_loop_model, model_name):
    """
    Get R2-score, MAE and MAPE
    """
    training_score = busy_loop_model.score(X_train, y_train)
    test_score = busy_loop_model.score(X_test, y_test)
    training_preds = busy_loop_model.predict(X_train)
    test_preds = busy_loop_model.predict(X_test)
    training_mae = round(mean_absolute_error(training_preds, y_train), 3)
    training_mape = round(mean_absolute_percentage_error(training_preds, y_train), 3)
    test_mae = round(mean_absolute_error(test_preds, y_test), 3)
    test_mape = round(mean_absolute_percentage_error(test_preds, y_test), 3)
    caption = f"##################### {model_name}-busy_loop ############################"
    print (caption)
    print(f"LR Model Train Score: {training_score}")
    print(f"LR Model Train MAE: {training_mae}")
    print(f"LR Model Train MAPE: {training_mape}")

    print(f"LR Model Test Score: {test_score}")
    print(f"LR Model Test MAE: {test_mae}")
    print(f"LR Model Test MAPE: {test_mape}")


models = ["Qwen/Qwen2.5-7B", "Qwen/Qwen3-14B"]
for model in models:
    model_name = model.split("/")[-1].replace(".", "_")
    train_df = processed_data(model_name, "train")
    test_df = processed_data(model_name, "test")
    X_train = train_df.loc[:, ~train_df.columns.isin(["busy_loop_time_with_req"])]
    y_train = train_df["busy_loop_time_with_req"]
    X_test = test_df.loc[:, ~test_df.columns.isin(["busy_loop_time_with_req"])]
    y_test = test_df["busy_loop_time_with_req"]
    model_lr = LinearRegression(positive=True)
    model_lr.fit(X_train, y_train)
    calculate_metrics(X_train, y_train, X_test, y_test, model_lr, model_name)
    
