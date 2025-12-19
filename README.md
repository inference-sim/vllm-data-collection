# vLLM Benchmark Data Collector

The scripts in this folder are for training and testing BLIS (https://github.com/inference-sim/inference-sim). This repository focuses on making the process of benchmarking vLLM and data collection straightforward. In terms of workloads and BLIS capabilities, the evolution of BLIS has been split into multiple scenarios. The folders `scenario1` to `scenario6` capture the differences in workloads used to train and test BLIS.

The instructions below mainly focus on training and testing BLIS for the following three scenarios:

| Scenario | vLLM Features | Workload | Prompt data source | Results Used (Metrics) |
| :--- | :--- | :--- | :--- |
| **Scenario 5** | `stream=False` | Concurrent requests, GuideLLM-style workload | Synthetic | Traces, GuideLLM results |
| **Scenario 5++** | `stream=True` | Concurrent requests, GuideLLM-style workload | Synthetic | Traces, GuideLLM results |
| **Scenario 6** | `stream=True` | Concurrent requests, GuideLLM-style workload | RedHat GuideLLM data | GuideLLM results | 

You can also look into directories `scenario1` to `scenario4` to get an understanding of the preceding scenarios that evolved eventually into these scenarios.

# Scenario5++:

> These instructions assume that you already have pretrained benchmark data from vLLM in the folder `train/` that contains the following files:
* `vllm.log`: vLLM server logs.
* `exp-config.yaml`: Benchmark specifications, containing mainly vllm server args (model, tp etc.) and `total-kv-blocks`. `total-kv-blocks` is currently parsed from `vllm.log` but you can also estimate it using Capacity Planner.
* `guidellm-results.json`: GuideLLM output file containing client-side benchmark metrics
* `profile.yaml`: Workload profile in GuideLLM format
* `traces.json`: vLLM server-side traces

For test, you should have similar benchmark data saved under the folder `test/`.

## Build BLIS

```
git clone git@github.com:inference-sim/inference-sim.git
go build -o simulation_worker main.go
```

## Train BLIS

To train both BLIS alpha and beta models for all benchmarks you have saved under `train/`:

`python train_blis.py --train_results_path train/ --model_path models/`

This gives you BLIS alpha and beta coefficients saved under `models/` to be used for testing/inference.

## Test BLIS

`python test_blis.py --test_results_path test/ --model_path models/ --groupby_field tp/chunk_size/app/rps/model_path`

Under `test_plots/blis`, you will find barplots showing average errors grouped by the field specified in the args.

# Scenario6:

> These instructions assume that you already have the pretrained RedHat GuideLLM data file, e.g: `blis_rh_final.xlsx`. If you do not have this file, please contact us for access.

## Build BLIS

```
git clone git@github.com:inference-sim/inference-sim.git
go build -o simulation_worker main.go
```

## Train BLIS

There are two ways you can train BLIS for Scenario6:

* Pick preset (LLM, TP, GPU, vllm-version) combinations from a `specs.csv` file. 
This trains each combination in the file.

```
python train_blis.py --specs-filepath specs.csv
```

* Train over a specific (LLM, TP, GPU, vllm-version) combination:

```
python train_blis.py --LLM_name ... --tp 1 --GPU H100 --vllm-version ...
```

By default the following files are picked by the training script. You can also pass these flags as args to the training script to provide custom values:

* `training-filepath` - Path to the RH GuideLLM data to train BLIS over. Default: `blis_rh_final.xlsx`
* `coeffs-filepath` - Path to save the trained BLIS coefficients to. Default: `coefficients.yaml`
* `specs-filepath` - Path to the CSV file containing all combinations of (LLM, TP, GPU, vllm-version) to train over. Default: `training_specs.csv`.

Note: If you rerun the exact same (LLM, TP, GPU, vllm-version) multiple times, the script simply overwrites existing coefficients in coeffs-filepath.

### FAQs:

* How do I change the cost function for the optimizer?

For this, go to the `train_blis.py` script and modify the constant `METRICS_IN_LOSS` list to include metrics in the cost function. Note that these metric names must match the column names in the GuideLLM Excel file.

* How do I run optimization over more/less iterations?

Similar to above, go to `train_blis.py` and update the constant `NUM_TPE_ITERS` to the number of iterations you want.

## Test BLIS

When you want to test BLIS's performance, you might want to group your error numbers by various fields. Currently, you can see average results across **all** "Test" row in the GuideLLM RH `.xlsx` file (provided you have pretrained coefficients for the relevant (LLM, TP, GPU, vllm-version) combination). You can also pass your custom `coeffs-filepath` and `training-filepath` (defaults are similar to the train script):

```
python test_blis.py
```

Or with custom `coeffs-filepath`:

```
python test_blis.py --coeffs-filepath coefficients.yaml
```

If you want to groupby LLM, TP, GPU or vllm-version, simply pass the argument to the test script. For example, the command below will group your test results by the `LLM-name == ibm-granite/granite-3.1-8b-instruct`:

```
python test_blis.py --LLM-name ibm-granite/granite-3.1-8b-instruct
```

The test error plots will be saved to a folder `test_plots/blis` in your current working directory. Depending on what arguments you provided for groupby, the plots will be saved as:

```
LLM=* TP=* GPU=* vllmVersion=*_error.png
LLM=granite-3.1-8b-instruct TP=* GPU=* vllmVersion=*_error.png
```

## Generate BLIS-simulated JSON file for [Compass](https://github.com/redhat-et/compass/blob/main/data/benchmarks.json):

To generate the final formatted synthetic metrics file, we currently take all the (LLM, TP, GPU, vllm-version) in the "Test" rows in the GuideLLM RH `.xlsx` file. We only take those (LLM, TP, GPU, vllm-version) combinations which have pretrained coefficients saved in the coeffs file. You can pass the config file and GuideLLM data filepath of your choice to the script. You can also pass the filename of the final synthetic metrics file to generate, as follows:

```
python generate_inference_json.py --coeffs-filepath coefficients.yaml --testing-filepath blis_rh_final.xlsx --synthetic-results-filepath benchmarks_BLIS.json
```