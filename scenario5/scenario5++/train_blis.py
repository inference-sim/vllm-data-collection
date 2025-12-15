import os
import argparse
import glob
import yaml
import sys

from postprocess_guidellm_common import perform_postprocessing_common
from postprocess_blis import perform_postprocessing_blis
from blis_alpha_model import train_alpha_model
from blis_beta_model import train_beta_model

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Read and parse traces JSON file.")
    parser.add_argument("--model_path", 
                        help="Path to trained beta models folder")
    parser.add_argument("--train_results_path", 
                        help="Path to the root train results folder")
    args = parser.parse_args()

    # Postprocess and train across all train experiments
    train_results_path = os.path.join(args.train_results_path, "*/")
    all_train = glob.glob(train_results_path)
    train_alpha_model(all_train, args.model_path)
    idx = 0
    for train_specific_path in all_train:
        if idx > 0:
            break
        guidellm_profile_path = os.path.join(train_specific_path, "profile.yaml")
        guidellm_results_path = os.path.join(train_specific_path, "guidellm-results.json")
        vllm_config_path = os.path.join(train_specific_path, "exp-config.yaml")

        traces_path = os.path.join(train_specific_path, "traces.json")
        perform_postprocessing_common(guidellm_results_path, train_specific_path)
        perform_postprocessing_blis(guidellm_profile_path, traces_path, vllm_config_path, train_specific_path, train=True)
        # get TP value from vllm_config
        try:
            with open(vllm_config_path, 'r') as f:
                vllm_config = yaml.safe_load(f)
                model = vllm_config["model"].split("/")[1].lower()
                tp = vllm_config["tensor_parallelism"]
        except:
            print("Could not load vllm config data.")
            sys.exit()

        # train and  save BLIS coeffs
        model_tp_path = os.path.join(args.model_path, f"model_{model}_tp_{tp}")
        os.makedirs(model_tp_path, exist_ok=True)
        train_beta_model(train_specific_path, args.model_path, model_tp_path)
        idx += 1