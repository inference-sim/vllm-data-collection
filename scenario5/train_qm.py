import os
import argparse
import glob

from postprocess_guidellm_common import perform_postprocessing_common
from postprocess_qm import perform_postprocessing_qm

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Read and parse traces JSON file.")
    parser.add_argument("--model_path", 
                        help="Path to trained beta models folder")
    parser.add_argument("--train_results_path", 
                        help="Path to the root test results folder")
    args = parser.parse_args()

    # Postprocess and train across all train experiments
    train_results_path = os.path.join(args.train_results_path, "*/")
    all_train = glob.glob(train_results_path)
    for train_path in all_train:
        guidellm_profile_path = os.path.join(train_path, "profile.yaml")
        guidellm_results_path = os.path.join(train_path, "guidellm-results.json")
        vllm_config_path = os.path.join(train_path, "exp-config.yaml")

        traces_path = os.path.join(train_path, "traces.json")
        perform_postprocessing_common(guidellm_results_path, train_path)
        perform_postprocessing_qm(traces_path, vllm_config_path, train_path, train=True)