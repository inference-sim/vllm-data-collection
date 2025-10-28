import json
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.inspection import permutation_importance
from collections import defaultdict
from experiment_configs_constants_train import *

NUM_GROUPS = 1000 #maybe should be 16k

def calculate_smape(actual, predicted) -> float:
    if not all([isinstance(actual, np.ndarray), isinstance(predicted, np.ndarray)]):
        actual, predicted = np.array(actual), np.array(predicted)

    return round(np.mean(np.abs(predicted - actual) / ((np.abs(predicted) + np.abs(actual))/2)), 2)

def get_request_info_per_step(model_name, mode, mbnt, rr, spec_small):
    request_level_step_info = defaultdict(list)
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
                                if event["event_type"] == "PREFIX":
                                    prompt_data["prefix"] = event["step"]
                            num_prefill_steps = int((prompt["input_len"] - prompt_data["prefix"] + mbnt - 1)//mbnt)
                            total_prefilled = int(prompt_data["prefix"])
                            for step in range(prompt_data["scheduled_step"], prompt_data["finished_step"] + 1): #all steps current req is running for
                                req_step_count = step - prompt_data["scheduled_step"] + 1
                                if req_step_count <= num_prefill_steps:
                                    num_cache_hits_for_req = total_prefilled
                                    num_cache_miss_for_req = min(mbnt, prompt["input_len"] - total_prefilled)
                                    total_prefilled += num_cache_miss_for_req
                                    request_level_step_info[str(step)].append([num_cache_hits_for_req, num_cache_miss_for_req]) #[hit, miss]
                                else:
                                    num_cache_hits_for_req = prompt["input_len"] + (req_step_count - num_prefill_steps - 1)
                                    request_level_step_info[str(step)].append([num_cache_hits_for_req, 0]) #[hit, miss]
    return request_level_step_info

def get_regression_features_from_step(per_step_data, y_column):
    features = {}
    features["beta_1"] = per_step_data["num_cache_hit_tokens"]
    features["beta_2"] = per_step_data["num_cache_miss_tokens"]
    beta_3_term = 0
    for req in per_step_data["requests"]: #req order is [hit, miss]
        beta_3_term += (req[0] * req[1] + req[1] * req[1])
    features["beta_3"] = beta_3_term
    features["beta_4"] = per_step_data["num_decode_reqs"]
    features["beta_5"] = per_step_data["num_finished_reqs"]
    features["beta_6"] = per_step_data["num_finished_tokens"]
    features["intercept"] = 1
    features[y_column] = per_step_data["loop_time"]
    
    return features

def get_fixed_cost_features_from_step(per_step_data, y_column):
    features = {}
    features["gamma_1"] = 1 if per_step_data["num_cache_miss_tokens"] > 0 else 0
    features["gamma_2"] = 1 if per_step_data["num_decode_reqs"] > 0 else 0
    features[y_column] = per_step_data["loop_time"]
    return features

def get_linear_cost_features_from_step(per_step_data, y_column):
    features = {}
    features["gamma_1"] = per_step_data["num_cache_miss_tokens"]
    features["gamma_2"] = per_step_data["num_decode_reqs"]
    features["intercept"] = 1
    features[y_column] = per_step_data["loop_time"]
    return features

def read_metrics_file(file_path, y_column, spec, mbnt):
    step_data = []
    try:
        with open(file_path, "r") as f:
            metrics_data = json.load(f)
            for step in metrics_data:
                if int(step) > 2500:
                    step_features = get_linear_cost_features_from_step(metrics_data[step], y_column)
                    step_features.update({"spec": spec, "mbnt": mbnt})
                    step_data.append(step_features)
    except:
        print(f"Cannot open {file_path}.")
    return step_data

def get_step_compositions(all_steps, start_step, end_step):
    agg_step_features = {"gamma_1": 0, "gamma_2": 0, "intercept": 0, "busy_loop_time_with_req": 0}
    # agg_step_features = {"beta_1": 0, "beta_2": 0, "beta_3": 0, "beta_4": 0, "beta_5": 0, "beta_6": 0, "intercept": 0, "busy_loop_time_with_req": 0}
    for step in range(start_step, end_step + 1):
        try:
            curr_step_features = get_linear_cost_features_from_step(all_steps[str(step)], "busy_loop_time_with_req")
            for key in agg_step_features:
                agg_step_features[key] += curr_step_features[key]
        except:
            pass # don't have the last chunk of steps
    return agg_step_features

def get_model_mode_latencies_by_req(model_name, mode, rr, spec):
    model_mode_data = []
    for mbnt in MAX_NUM_BATCHED_TOKENS:
        spec_small = spec.lower()
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
                                    if event["event_type"] == "SCHEDULED": # take the last scheduled event
                                        scheduled_time = event["timestamp"]
                                        prompt_data["scheduled_step"] = event["step"]
                                    # if event["event_type"] == "PREFIX": # take the last prefix hit
                                    #     prompt_data["prefix"] = event["step"]
                                prompt_data["exp_path"] = dirpath
                                prompt_data["spec"] = spec
                                prompt_data["mbnt"] = mbnt
                                model_mode_data.append(prompt_data) #only append if no error
    return model_mode_data

def combine_metrics_jsons(mode, exp_path, filenames):
    all_steps = {}
    for filename in filenames:
        if filename.startswith(f"metrics_{mode}"):
            full_path = os.path.join(exp_path, filename)
            try:
                with open(full_path, "r") as f:
                    metrics_data = json.load(f)
                    all_steps.update(metrics_data)
            except:
                print(f"Cannot read {full_path}")
    return all_steps

def processed_data_by_req(model_name, mode, rr, spec):
    model_mode_latencies = get_model_mode_latencies_by_req(model_name, mode, rr, spec)
    request_df = pd.DataFrame(model_mode_latencies)
    all_step_request_merged_dfs = []
    for mbnt in MAX_NUM_BATCHED_TOKENS:
        spec_small = spec.lower()
        results_folder = f"../results_new/scenario4/{model_name}/{mode}/{spec_small}/mbnt_{mbnt}/rr_{rr}"
        for dirpath, _, filenames in os.walk(results_folder):
            if dirpath.endswith(f"{mode}_scenario4"):
                all_steps = combine_metrics_jsons(mode, dirpath, filenames)
                request_df_curr = request_df[(request_df["exp_path"]==dirpath) & (request_df["spec"]==spec) & (request_df["mbnt"]==mbnt)]
                request_df_curr = request_df_curr[["exp_path", "scheduled_step", "finished_step", "spec", "mbnt", "input_len", "output_len"]]
                request_df_curr["steps"] = request_df_curr.apply(lambda x: get_step_compositions(all_steps, x["scheduled_step"], x["finished_step"]), axis = 1)
                expanded_cols = request_df_curr['steps'].apply(pd.Series)
                request_df_curr = request_df_curr.join(expanded_cols)
                all_step_request_merged_dfs.append(request_df_curr)
    return pd.concat(all_step_request_merged_dfs, ignore_index=True)

def processed_data_by_step(model_name, mode, rr, spec):
    step_data = []
    for mbnt in MAX_NUM_BATCHED_TOKENS:
        spec_small = spec.lower()
        results_folder = f"../results_new/scenario4/{model_name}/{mode}/{spec_small}/mbnt_{mbnt}/rr_{rr}"
        if os.path.isdir(results_folder):
            for dirpath, _, filenames in os.walk(results_folder):
                for filename in filenames:
                    if filename.startswith(f"metrics_{mode}"):
                        full_path = os.path.join(dirpath, filename)
                        step_data.extend(read_metrics_file(full_path, "loop_latency", spec, mbnt))

    step_df = pd.DataFrame(step_data)    
    step_df = step_df.dropna()
    return step_df

def create_step_groups(step_df, rr):
    """
    Efficiently filters a DataFrame using vectorized boolean masking.
    """
    all_groups = []
    for _ in range(NUM_GROUPS):
        if rr == 5:
            p = np.random.uniform(0, 1)/10
        else:
            p = np.random.uniform(0, 1)
        group_df = step_df[np.random.rand(len(step_df)) < p]
        all_groups.append(group_df.sum())
    return pd.DataFrame(all_groups)

def calculate_metrics(X_train, y_train, X_test, y_test, busy_loop_model, model_name, include_val = False):
    """
    Get R2-score, MAE and MAPE
    """
    training_score = busy_loop_model.score(X_train, y_train)
    training_preds = busy_loop_model.predict(X_train)
    training_mae = round(mean_absolute_error(training_preds, y_train), 3)
    training_mape = round(mean_absolute_percentage_error(training_preds, y_train), 3)
    training_smape = round(calculate_smape(training_preds, y_train), 3)
    if include_val:
        test_preds = busy_loop_model.predict(X_test)
        test_score = busy_loop_model.score(X_test, y_test)
        test_mae = round(mean_absolute_error(test_preds, y_test), 3)
        test_mape = round(mean_absolute_percentage_error(test_preds, y_test), 3)
        test_smape = round(calculate_smape(test_preds, y_test), 3)
    caption = f"##################### {model_name}-busy_loop ############################"
    print(caption)
    print(f"LR Model Train Score: {training_score}")
    print(f"LR Model Train MAE: {training_mae}")
    print(f"LR Model Train MAPE: {training_mape}")
    print(f"LR Model Train SMAPE: {training_smape}")
    coeffs = {}
    for idx, feature in enumerate(X_train.columns):
        coeffs[feature] = busy_loop_model.coef_[idx]
    print(f"Coeffs: {float(coeffs["intercept"]), float(coeffs["gamma_1"]), float(coeffs["gamma_2"])}")

    if include_val:
        print(f"LR Model Test Score: {test_score}")
        print(f"LR Model Test MAE: {test_mae}")
        print(f"LR Model Test MAPE: {test_mape}")
        print(f"LR Model Test SMAPE: {test_smape}")

        feature_imp = permutation_importance(busy_loop_model, X_test, y_test, n_repeats=10, random_state=42)
        print("Feature Importances (Mean):")
        for i, importance in enumerate(feature_imp.importances_mean):
            print(f"{X_test.columns[i]}: {importance:.4f}")
        print("\nFeature Importances (Standard Deviation):")
        for i, std_dev in enumerate(feature_imp.importances_std):
            print(f"{X_test.columns[i]}: {std_dev:.4f}")

def populate_steps_with_req_level_hits(model_name, mode, rr, spec):
    for mbnt in MAX_NUM_BATCHED_TOKENS:
        spec_small = spec.lower()
        request_level_step_info = get_request_info_per_step(model_name, mode, mbnt, rr, spec_small)
        results_folder = f"../results_new/scenario4/{model_name}/{mode}/{spec_small}/mbnt_{mbnt}/rr_{rr}"
        if os.path.isdir(results_folder):
            for dirpath, _, filenames in os.walk(results_folder):
                for filename in filenames:
                    if filename.startswith(f"metrics_{mode}"):
                        full_path = os.path.join(dirpath, filename)
                        print(f"Preprocessing {full_path}...")
                        try:
                            with open(full_path, "r") as f:
                                metrics_data = json.load(f)
                                for step in metrics_data:
                                    if len(request_level_step_info[step]) > 0:
                                        metrics_data[step]['requests'] = request_level_step_info[step]
                                    else:
                                        metrics_data[step]['requests'] = []
                            with open(full_path, 'w') as f:
                                json.dump(metrics_data, f, indent=4)
                        except:
                            pass


def train_requestwise_test_requestwise(train_configs, val_configs, include_val = False):
    model_name = MODEL.split("/")[-1].replace(".", "_")
    group_train_dfs = []
    for config in train_configs:
        train_df = processed_data_by_req(model_name, "train", config["rr"], config["spec"])
        train_df = train_df.drop(columns=["exp_path", "scheduled_step", "finished_step", "steps", "spec", "mbnt", "input_len", "output_len"])
        group_train_dfs.append(train_df)
    group_train_df = pd.concat(group_train_dfs, ignore_index=True)
    group_train_df = group_train_df.dropna()
    group_train_df = group_train_df[group_train_df['busy_loop_time_with_req'] > 0]
    X_train = group_train_df.loc[:, ~group_train_df.columns.isin(["busy_loop_time_with_req"])]
    y_train = group_train_df["busy_loop_time_with_req"]
    model_lr = LinearRegression(positive=True, fit_intercept=False)
    model_lr.fit(X_train, y_train)
    if not include_val:
        calculate_metrics(X_train, y_train, [], [], model_lr, model_name, include_val)
    else:
        # validation is grouped by spec for easier interpretability
        val_specs = list(set(x['spec'] for x in val_configs))
        for config in val_configs:
            val_df = processed_data_by_req(model_name, "val", config["rr"], config["spec"])
            val_df = val_df[val_df['busy_loop_time_with_req'] > 0]
            for spec in val_specs:
                val_df_spec = val_df[val_df["spec"] == spec]
                val_df_spec = val_df_spec.drop(columns=["exp_path", "scheduled_step", "finished_step", "steps", "spec", "mbnt", "input_len", "output_len"])
                X_test = val_df_spec.loc[:, ~val_df_spec.columns.isin(["busy_loop_time_with_req"])]
                y_test = val_df_spec["busy_loop_time_with_req"]
                if X_test.shape[0] > 0:
                    calculate_metrics(X_train, y_train, X_test, y_test, model_lr, model_name, include_val)

def train_groupwise_test_groupwise(train_configs, val_configs, include_val = False):
    model_name = MODEL.split("/")[-1].replace(".", "_")
    group_train_dfs = []
    for config in train_configs:
        train_df = processed_data_by_step(model_name, "train", config["rr"], config["spec"])
        train_df = train_df.drop(columns=["spec", "chunk_size", "prefix_ratio"]) # do not drop input and output len
        group_train_dfs.append(create_step_groups(train_df, config["rr"]))
    group_train_df = pd.concat(group_train_dfs, ignore_index=True)
    group_train_df = group_train_df.dropna()
    group_train_df = group_train_df[group_train_df['loop_latency'] > 0]
    X_train = group_train_df.loc[:, ~group_train_df.columns.isin(["loop_latency"])]
    y_train = group_train_df["loop_latency"]
    model_lr = LinearRegression(positive=True, fit_intercept=False)
    model_lr.fit(X_train, y_train)
    # validation is grouped by spec for easier interpretability
    val_specs = list(set(x['spec'] for x in val_configs))
    for config in val_configs:
        test_df = create_step_groups(processed_data_by_step(model_name, "test", config["rr"], config["spec"]), config["rr"])
        # test_df = test_df.drop(columns=["exp_path", "scheduled_step", "finished_step", "steps", "spec", "chunk_size", "prefix_ratio", "input_len", "output_len"])
        test_df = test_df[test_df['loop_latency'] > 0]
        for spec in val_specs:
            test_df_spec = test_df[test_df["spec"] == spec]
            test_df_spec = test_df_spec.drop(columns=["spec", "chunk_size", "prefix_ratio"]) # dropped input and output len
            X_test = test_df_spec.loc[:, ~test_df_spec.columns.isin(["loop_latency"])]
            y_test = test_df_spec["loop_latency"]
            if X_test.shape[0] > 0:
                calculate_metrics(X_train, y_train, X_test, y_test, model_lr, model_name, include_val)
            else:
                print("Empty test group")

if __name__=="__main__": 
    np.random.seed(42)

    # find non-saturated training regimes from postprocess_vllm's outputs
    train_configs = []
    model_name = MODEL.split("/")[-1].replace(".", "_")
    results_folder = f"results_server_side/{model_name}/train"
    if os.path.isdir(results_folder):
        for dirpath, _, filenames in os.walk(results_folder):
            for filename in filenames:
                rr = filename.split("_")[1][:-1]
                spec = filename.split("_")[2]
                train_configs.append({"rr": rr, "spec": spec})
    val_configs = []

    # use for quadratic features only
    # modes = ["train", "test"]
    # for model in models:
    #     model_name = model.split("/")[-1].replace(".", "_")
    #     for mode in modes:
    #         populate_steps_with_req_level_hits(model_name, mode)
    train_requestwise_test_requestwise(train_configs, val_configs, include_val=False)