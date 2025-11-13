import argparse
import json
import os
import sys

def get_GuideLLM_rps_list(guidellm_results):
    """
    Read GuideLLM results file and extract list of
    uniformly spaced constant RPS values.
    Returns:

    rps_list: List of constant RPS values in GuideLLM benchmark
    """
    rps_list = []
    profile = guidellm_results["benchmarks"][0]["config"]["profile"]
    for strategies in profile["completed_strategies"]:
        if strategies["type_"] == "constant":
            rps_list.append(float(strategies["rate"]))
    return rps_list

def get_sweep_info(guidellm_results, rps_list):
    """
    Get details about each GuideLLM sweep trial (unique RPS).
    Details include: constant rps value and response-ids (vLLM requestIDs)
    """
    sweep_info = []
    for rps_idx, rps in enumerate(rps_list):
        current_sweep = {}
        current_sweep["rps"] = rps
        current_sweep["requestIDs"] = []
        # exclude synchronous(idx: 0) and throughput(idx: 1)
        all_requests = guidellm_results["benchmarks"][rps_idx + 2]["requests"]
        for req in all_requests["successful"]:
            current_sweep["requestIDs"].append(req["response_id"])
        sweep_info.append(current_sweep)
    return sweep_info


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Read and parse traces JSON file.")
    parser.add_argument("--guidellm_results_file", 
                        help="Path to the GuideLLM JSON output file to be read.")
    parser.add_argument("--sweep_info_path",
                        default=".", 
                        help="Location to save GuideLLM's sweep trial info")
    args = parser.parse_args()

    guidellm_results_filepath = args.guidellm_results_file
    sweep_info_filepath = os.path.join(args.sweep_info_path, "sweep_info.json")

    # read GuideLLM results file
    try:
        with open(guidellm_results_filepath, 'r') as f:
            guidellm_results = json.load(f)
    except:
        print("Could not read GuideLLM results file.")
        sys.exit()

    # Get list of constant request rates (RPS)
    rps_list = get_GuideLLM_rps_list(guidellm_results)

    # Get sweep trial info - RPS, vLLM-assigned requestIDs
    sweep_info = get_sweep_info(guidellm_results, rps_list)

    # Save sweep info to JSON file
    with open(sweep_info_filepath, 'w+') as f:
        json.dump(sweep_info, f, indent=4)