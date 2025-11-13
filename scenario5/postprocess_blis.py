import argparse
import json
import os
import sys
import yaml

def construct_BLIS_reqgenconfig(guidellm_profile, rps):
    """
    Given GuideLLM profile and request rates,
    construct BLIS-style request gen-config file.
    """
    blis_reqgen_config = {
        "format": "GuideLLM",
        "seed": guidellm_profile["random-seed"], 
        "rate": []
    }
    blis_reqgen_config["rate"] = {
        "arrival-type": "Constant",
        "rate": rps,
        "max-requests": guidellm_profile["max-requests"]
    }
    blis_reqgen_config["data"] = guidellm_profile["data"]
    return blis_reqgen_config

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Read and parse traces JSON file.")
    parser.add_argument("--guidellm_profile_file", 
                        help="Path to the GuideLLM YAML profile file to be read.")
    parser.add_argument("--sweep_info_path",
                        default="sweep_info.json",
                        help="Path to GuideLLM's sweep trial info")
    parser.add_argument("--blis_reqgen_config_folder",
                        default="blis_reqgenconfigs",
                        help="Folder to save BLIS' YAML request gen-config file")
    
    args = parser.parse_args()

    # read GuideLLM sweep info
    with open(args.sweep_info_path, 'r') as f:
        sweep_info = json.load(f)

    guidellm_profile_filepath = args.guidellm_profile_file
    # read GuideLLM profile file
    try:
        with open(guidellm_profile_filepath, 'r') as f:
            guidellm_profile = yaml.safe_load(f)
    except:
        print("Could not read GuideLLM profile file.")
        sys.exit()
    
    # Check if args.blis_reqgen_config_folder exists, otherwise create
    os.makedirs(args.blis_reqgen_config_folder, exist_ok=True)

    # Construct BLIS-style request gen-config
    for sweep in sweep_info:
        rps = sweep["rps"]
        blis_reqgen_config = construct_BLIS_reqgenconfig(guidellm_profile, rps)
        blis_reqgen_config_filename = os.path.join(
            args.blis_reqgen_config_folder, 
            f"requestgenconfig_RPS={round(rps, 3)}.yaml"
        )

        # Save to YAML file - one per RPS
        with open(blis_reqgen_config_filename, 'w+') as f:
            yaml.dump(blis_reqgen_config, f)
