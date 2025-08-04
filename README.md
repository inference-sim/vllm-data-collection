# vLLM Benchmark Data Collector

The scripts in this folder are for gathering performance data from vLLM. This data can be used for various purposes, including building a vLLM simulator. This repository focuses on making the process of benchmarking and data collection straightforward.

## Setup

1. Install vLLM:

For our vllm experiments we made edits to the source code, to gather data for our simulation.
That is no longer needed and the scripts of this folder can be used independent of vllm.

For latest installation: https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#create-a-new-python-environment

From source (python only edits):
``` bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

Compiled install (nvidia gpus) :
```bash
uv venv --python 3.12 --seed
source .venv/bin/activate

# Install vLLM with CUDA 12.8.
# If you are using pip.
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
# If you are using uv.
uv pip install vllm --torch-backend=auto
```

## Running Benchmarks

The `benchmarks/benchmark_runner_simulator.py` script is the main entrypoint for running the benchmark experiments. It takes a configuration file and an output directory as arguments.

```bash
python benchmarks/benchmark_runner_simulator.py <config_file.yaml> --output <output_directory>
```

The script orchestrates the benchmarking process as follows:
1.  It parses the specified configuration file, which contains a set of experiments to run.
2.  For each experiment, it performs the following steps:
    a.  **Start vLLM Server**: It programmatically starts a vLLM server instance with the parameters defined in the `vllm` section of the experiment's configuration.
    b.  **Wait for Server**: It waits until the vLLM server is up and ready to accept requests.
    c.  **Run Benchmark**: It then executes the benchmark against the server using the parameters from the `benchmark` section of the config.
3.  After each experiment is completed, the vLLM server is shut down. This ensures that each experiment runs in a clean, isolated environment, and all state is flushed between runs.

### Benchmark Output

For information on how data is outputted from the benchmark runner, see the documentation here: [Benchmark Output Format](./docs/output_format.md) (placeholder).

## Configuration

### Generating Configurations

A configuration file containing a sweep of experiments can be generated using the `benchmarks/config_generator.py` script.

```bash
python benchmarks/config_generator.py
```

This script creates a `vllm_benchmark_config.yaml` file in the `benchmarks` directory. It generates a comprehensive set of experiment configurations by creating a Cartesian product of all parameter lists defined in its `main()` function.

The generator includes logic to skip invalid or redundant parameter combinations. For example, it will skip scenarios where `long_prefill_token_threshold` is greater than `max_num_batched_tokens`, as this represents a non-unique behavior.

### Configuration Structure

Each generated configuration is structured into two main parts: `vllm` parameters and `benchmark` parameters.

-   `vllm`: Contains parameters for configuring the vLLM server itself.
-   `benchmark`: Contains parameters for the benchmarking client that sends requests to the server.

Here is an example of a single experiment configuration:

```yaml
  baseline:
    name: "exp1"
    description: "Basic vLLM performance test"
    model: "Qwen/Qwen2.5-0.5B"
    runs: 1

    # vLLM server parameters
    vllm:
      gpu_memory_utilization: 0.9
      enable_prefix_caching: true
      disable_log_requests: false
      block_size: 16 
      max_model_len: 2048
      max_num_batched_tokens: 2048
      max_num_seqs: 256
      long_prefill_token_threshold: 1000000
      seed: 42

    # Benchmark parameters
    benchmark:
      backend: "vllm"
      dataset_name: "sharegpt"
      dataset_path: "ShareGPT_V3_unfiltered_cleaned_split.json"
      num_prompts: 100
      request_rate: 16
      sharegpt_output_len: 0
      temperature: 0.0
      seed: 42
```

### Adding New Parameters

To add a new parameter to the sweep, you need to modify `benchmarks/config_generator.py`. Follow these steps:

1.  **Define Parameter Values**: In the `main()` function of `config_generator.py`, add a new list containing the values you want to sweep over for your new parameter.
    ```python
    # In main()
    my_new_param_list = [value1, value2, value3]
    ```

2.  **Update `generate_config` Call**: Pass the new list as an argument to the `generate_config` function call within `main()`.

3.  **Add to Parameter Product**: In the `generate_config` function, add your new parameter list to the function signature and to the `itertools.product()` call. This will ensure it's included in the parameter sweep.
    ```python
    # In generate_config()
    for ..., my_new_param in product(
        ..., my_new_param_list
    ):
        # ...
    ```

4.  **Place in Experiment Config**: Add the new parameter to the `exp_config` dictionary. Make sure to place it under the correct key (`vllm` for server parameters or `benchmark` for benchmark client parameters).
    ```python
    # In generate_config()
    exp_config = {
        # ...
        'vllm': {
            # ...
            'my_new_vllm_param': my_new_param,
        },
        'benchmark': {
            # ...
            'my_new_benchmark_param': my_new_param,
        }
    }
    ```

5.  **Verify**: Run `python benchmarks/config_generator.py` to generate the new configuration file and verify that your new parameter is included correctly in the experiments.
