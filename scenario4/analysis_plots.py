import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from scipy.stats import norm
import seaborn as sns
import numpy as np
from experiment_configs_constants_train import *

# TODO: Refector this script

def get_fixed_cost_prediction(per_step_data, y_column):
    features = {}
    features["gamma_1"] = 1 if per_step_data["num_cache_miss_tokens"] > 0 else 0
    features["gamma_2"] = 1 if per_step_data["num_decode_reqs"] > 0 else 0
    # features[y_column] = 0.05258597*features["gamma_1"] + 0.0030444*features["gamma_2"]
    features[y_column] = per_step_data["loop_time"]
    return features

def get_linear_cost_features_from_step(per_step_data, y_column):
    features = {}
    features["gamma_1"] = per_step_data["num_cache_miss_tokens"]
    features["gamma_2"] = per_step_data["num_decode_reqs"]
    features["intercept"] = 1
    features[y_column] = per_step_data["loop_time"]
    return features

def get_steplevel_features(per_step_data, y_column, spec, chunk_size):
    features = {}
    features["total_cache_miss_tokens"] = per_step_data["num_cache_miss_tokens"]
    features["num_decode_reqs"] = per_step_data["num_decode_reqs"]
    features["num_finished_tokens"] = per_step_data["num_finished_tokens"]
    features["num_finished_reqs"] = per_step_data["num_finished_reqs"]
    features[y_column] = per_step_data["loop_time"]
    features["spec"] = spec
    features["chunk_size"] = chunk_size
    return features

def get_step_compositions(all_steps, start_step, end_step, output_len, mode):
    agg_step_features = {"gamma_1": 0, "gamma_2": 0, "intercept": 0, "busy_loop_time_with_req": 0}
    if mode == "prefill":
        end_step = max(start_step, end_step - output_len)
    elif mode == "decode":
        start_step = end_step - output_len + 1
    for step in range(start_step, end_step + 1):
        try:
            curr_step_features = get_linear_cost_features_from_step(all_steps[str(step)], "busy_loop_time_with_req")
            for key in agg_step_features:
                agg_step_features[key] += curr_step_features[key]
        except Exception as e:
            pass
    return agg_step_features

def get_model_mode_latencies_by_req(model_name, mode, rr):
    model_mode_data = []
    for spec in specs:
        for chunk_size in CHUNK_SIZES:
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
                                    # prompt_data["busy_loop_time_with_req"] = finished_time - scheduled_time
                                    # prompt_data["busy_loop_time_waiting"] = scheduled_time - queued_time
                                    prompt_data["exp_path"] = dirpath
                                    prompt_data["spec"] = spec
                                    prompt_data["chunk_size"] = chunk_size
                                    model_mode_data.append(prompt_data) #only append if no error
    return model_mode_data

def combine_metrics_jsons(mode, exp_path, filenames):
    all_steps = {}
    for filename in filenames:
        if filename.startswith(f"metrics_{mode}"):
            full_path = os.path.join(exp_path, filename)
            # print(f"Preprocessing {full_path}...")
            try:
                with open(full_path, "r") as f:
                    metrics_data = json.load(f)
                    all_steps.update(metrics_data)
            except:
                print(f"Cannot read {full_path}")
    return all_steps

def processed_data_by_req(model_name, mode, rr, prefill_or_decode):
    model_mode_latencies = get_model_mode_latencies_by_req(model_name, mode, rr)
    request_df = pd.DataFrame(model_mode_latencies)
    all_step_request_merged_dfs = []
    for spec in specs:
        for chunk_size in CHUNK_SIZES:
            spec_small = spec.lower()
            results_folder = f"../results_new/scenario4/{model_name}/{spec_small}/chunk_size_{chunk_size}/rr_{rr}"
            for dirpath, dirnames, filenames in os.walk(results_folder):
                if dirpath.endswith("scenario4"):
                    all_steps = combine_metrics_jsons(mode, dirpath, filenames)
                    request_df_curr = request_df[(request_df["exp_path"]==dirpath) & (request_df["spec"]==spec) & (request_df["chunk_size"]==chunk_size) & (request_df["prefix_ratio"]==prefix_hit_ratio)]
                    request_df_curr = request_df_curr[["exp_path", "scheduled_step", "finished_step", "spec", "chunk_size", "prefix_ratio", "input_len", "output_len"]]
                    request_df_curr[f"{prefill_or_decode}_steps"] = request_df_curr.apply(lambda x: get_step_compositions(all_steps, x["scheduled_step"], x["finished_step"], x["output_len"], prefill_or_decode), axis = 1)
                    # request_df_curr["decode_steps"] = request_df_curr.apply(lambda x: get_step_compositions(all_steps, x["scheduled_step"], x["finished_step"], x["output_len"], "decode"), axis = 1)
                    expanded_cols = request_df_curr[f"{prefill_or_decode}_steps"].apply(pd.Series)
                    request_df_curr = request_df_curr.join(expanded_cols)
                    all_step_request_merged_dfs.append(request_df_curr)
    return pd.concat(all_step_request_merged_dfs, ignore_index=True)

def plot_requestlevel_loop_times(df, model_name, rr, prefill_or_decode):
    df = df.drop(columns = ["exp_path", f"{prefill_or_decode}_steps"])

    grouped = df.groupby(['spec', 'chunk_size', 'prefix_ratio'])
    grouped.count().to_csv(f"rr_{rr}_df.csv")

    for (spec, chunk_size, prefix_ratio), group_df in grouped:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))
        label = f'Spec={spec}, Chunk Size={chunk_size}, Prefix Ratio={prefix_ratio}'
        
        if prefill_or_decode == "prefill":
            ax.scatter(group_df['input_len'], group_df['busy_loop_time_with_req'], marker='o', linestyle='-', label=label)
            ax.set_title(f'Prefill Busy Loop Ground-Truth Time for rr={rr}, chunk={chunk_size}, spec={spec}, pf={prefix_ratio}', fontsize=16)
            ax.set_xlabel('Input len', fontsize=12)
            ax.set_ylabel('Sum of prefill loop times (s)', fontsize=12)
        else:
            ax.scatter(group_df['output_len'], group_df['busy_loop_time_with_req'], marker='o', linestyle='-', label=label)
            ax.set_title(f'Decode Busy Loop Ground-Truth Time for rr={rr}, chunk={chunk_size}, spec={spec}, pf={prefix_ratio}', fontsize=16)
            ax.set_xlabel('Output len', fontsize=12)
            ax.set_ylabel('Sum of decode loop times (s)', fontsize=12)
        # ax.legend(title='Variants', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        dir_name = f"{model_name}_rr={rr}_{prefill_or_decode}_requestlevel"
        os.makedirs(dir_name, exist_ok=True)
        plt.savefig(f"{dir_name}/{model_name}_chunk={chunk_size}_spec={spec}_pf={prefix_ratio}_{prefill_or_decode}_vs_input_test.png")

    plt.close()

def plot_steplevel_loop_times(df, model_name, rr):
    grouped = df.groupby(['spec', 'chunk_size', 'prefix_ratio'])
    grouped.count().to_csv(f"rr_{rr}_df.csv")

    for (spec, chunk_size, prefix_ratio), group_df in grouped:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(2, 2, figsize=(30, 20))
        label = f'Spec={spec}, Chunk Size={chunk_size}, Prefix Ratio={prefix_ratio}'
        
        ax[0,0].scatter(group_df["total_cache_miss_tokens"], group_df['loop_latency'], marker='o', linestyle='-', label=label)
        ax[0,1].scatter(group_df["num_decode_reqs"], group_df['loop_latency'], marker='o', linestyle='-', label=label)
        ax[1,0].scatter(group_df["num_finished_tokens"], group_df['loop_latency'], marker='o', linestyle='-', label=label)
        ax[1,1].scatter(group_df["num_finished_reqs"], group_df['loop_latency'], marker='o', linestyle='-', label=label)

        for i in range(0, 2):
            for j in range(0, 2):
                ax[i,j].set_title(f'Busy Loop Time for rr={rr}, chunk={chunk_size}, spec={spec}, pf={prefix_ratio}', fontsize=16)
                ax[i,j].set_ylabel('Loop time', fontsize=12)
                # ax[i,j].legend(title='Variants', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax[0,0].set_xlabel('Total cache miss tokens', fontsize=12)
        ax[0,1].set_xlabel('Num decode reqs', fontsize=12)
        ax[1,0].set_xlabel('Num finished tokens', fontsize=12)
        ax[1,1].set_xlabel('Num finished reqs', fontsize=12)
        
        plt.tight_layout()
        dir_name = f"{model_name}_rr={rr}_steplevel"
        os.makedirs(dir_name, exist_ok=True)
        plt.savefig(f"{dir_name}/{model_name}_chunk={chunk_size}_spec={spec}_loop_times_test.png")

        plt.close()

        # df1 = group_df.loc[:,["total_cache_miss_tokens", "num_decode_reqs"]]
        # axs = sns.jointplot(x="total_cache_miss_tokens", y="num_decode_reqs", data=df1)
        # sns.distplot(df1.total_cache_miss_tokens, ax=axs.ax_marg_x, fit=norm)
        # sns.distplot(df1.num_decode_reqs, ax=axs.ax_marg_y, vertical=True, fit=norm)
    g = sns.JointGrid(data=df, y="total_cache_miss_tokens", x="num_decode_reqs")
    g.plot_joint(sns.scatterplot, s=100, alpha=.5)
    g.plot_marginals(sns.histplot, kde=True)
    g.fig.suptitle(f"Jointplot for rr={rr}")
        # axs.ax_joint.scatter("total_cache_miss_tokens", "num_decode_reqs", data=df1, c='r', marker='x')

    plt.savefig(f"{dir_name}/{model_name}_rr={rr}_jointplot.png")

def read_metrics_file(file_path, y_column, spec, chunk_size):
    step_data = []
    try:
        with open(file_path, "r") as f:
            metrics_data = json.load(f)
            for step in metrics_data:
                if int(step) > 2700:
                    step_data.append(get_steplevel_features(metrics_data[step], y_column, spec, chunk_size))
    except:
        print(f"Cannot open {file_path}.")
    return step_data

def processed_data_by_step(model_name, mode, request_rate):
    step_data = []
    for spec in specs:
        for chunk_size in CHUNK_SIZES:
            spec_small = spec.lower()
            results_folder = f"../results_new/scenario4/{model_name}/{spec_small}/chunk_size_{chunk_size}/rr_{request_rate}"
            if os.path.isdir(results_folder):
                for dirpath, _, filenames in os.walk(results_folder):
                    for filename in filenames:
                        if filename.startswith(f"metrics_{mode}"):
                            full_path = os.path.join(dirpath, filename)
                            step_data.extend(read_metrics_file(full_path, "loop_latency", spec, chunk_size))

    step_df = pd.DataFrame(step_data)
    step_df = step_df.dropna()
    return step_df

