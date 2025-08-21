import argparse
import yaml
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description='Simple vLLM Benchmark Runner')
parser.add_argument('--mode', help='train/test', default="train.yaml")
parser.add_argument('--model', help='LLM name', default="facebook/opt-125m")
args = parser.parse_args()

config_file = f"scenario1_config_{args.mode}.yaml"

filename = f"prompts_unique_{args.mode}.txt"

with open(config_file, "r") as f:
    config = yaml.safe_load(f) # read necessary configs and seed files
data_config = config["data"]
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
tokens = []
for i in range(97, 123):
    for j in range(97, 123):
        token = chr(i) + chr(j)
        tokens.append(token)

final_prompts = []
for idx1, input_len in enumerate(data_config["workload"]["input_lens"]):
    for idx2 in range(data_config["workload"]["num_exps"]):
        token_id = tokenizer.encode(token, add_special_tokens=False)
        if len(token_id) == 1:
            prompt = tokens[idx1 * data_config["workload"]["num_exps"] + idx2]
            while True:
                encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False)
                if len(encoded_prompt) == input_len - 1:
                    break
                prompt += " the"

            final_prompts.append(prompt)

with open(filename, "w") as f:
    for prompt in final_prompts:
        f.write(f"{prompt}\n")
print ("Successfully wrote tokens to: ", filename)