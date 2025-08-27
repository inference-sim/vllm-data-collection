import os
import subprocess

## Things to setup pvc-debugger pod for model download:
## pip install torch
## pip install transformers
## python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; m='MODEL_NAME'; AutoTokenizer.from_pretrained(m, trust_remote_code=True); AutoModelForCausalLM.from_pretrained(m, trust_remote_code=True)"

def copy_file_from_pod_using_kubectl_cp(pod_name, namespace, remote_dir_path, local_dir_path):
    try:
        command = [
            'oc', 'cp',
            f'{namespace}/{pod_name}:{remote_dir_path}',
            local_dir_path,
        ]
        print(command)
        result = subprocess.run(command, capture_output=True, text=True)
        print(result)

        if result.returncode == 0:
            print(f"'{remote_dir_path}' copied to '{local_dir_path}' successfully.")
            return True
        else:
            print(f"Error copying results: {result.stderr}")
            return False

    except Exception as e:
        print(f"Error executing oc cp command: {e}")
        return False

def main():
    pod_name = "pvc-debugger"
    namespace = "blis"
    local_dir_path = "results_new/"
    remote_dir_path = "/mnt/results/"
    os.makedirs(local_dir_path, exist_ok=True)
    copy_file_from_pod_using_kubectl_cp(pod_name, namespace, remote_dir_path, local_dir_path)

if __name__=="__main__":
    main()