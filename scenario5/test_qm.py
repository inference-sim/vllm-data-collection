import argparse
import json
import os
import glob
import sys
import yaml

from postprocess_qm import perform_postprocessing_qm    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Read and parse traces JSON file.")
    parser.add_argument("--test_results_path", 
                        help="Path to the root test results folder")
    args = parser.parse_args()

    # Evaluate across all test experiments
    test_results_path = os.path.join(args.test_results_path, "*/")
    all_tests = glob.glob(test_results_path)
    all_qm_data = {"TP2": [], "TP4": [], "TP8": []}
    for test_path in all_tests:
        vllm_config_path = os.path.join(test_path, "exp-config.yaml")
        # get TP value from vllm_config
        try:
            with open(vllm_config_path, 'r') as f:
                vllm_config = yaml.safe_load(f)
                tp = f"TP{vllm_config["tensor_parallelism"]}"
                chunk_size = vllm_config["max_num_batched_tokens"]
        except:
            print("Could not load vllm config data.")
            sys.exit()
        traces_path = os.path.join(test_path, "traces.json")
        qm_data = perform_postprocessing_qm(traces_path, vllm_config_path, test_path, train=False)
        all_qm_data[tp].extend(qm_data)
    for qm_data in all_qm_data:
        test_filename = os.path.join(args.test_results_path, f"QM_test_{qm_data}_{chunk_size}.json")
        with open(test_filename, 'w+') as f:
            json.dump(all_qm_data[qm_data], f, indent=4)
