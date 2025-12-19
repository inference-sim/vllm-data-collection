## Train Instructions


These instructions assume that you already have pretrained benchmark data from vLLM in the folder `train/` that contains the following files:

* `exp-config.yaml`: Benchmark specifications, containing mainly vllm server args (model, tp etc.) and `total-kv-blocks`. `total-kv-blocks` is currently estimated using Capacity Planner.
* `guidellm-results.json`: GuideLLM output file containing client-side benchmark metrics
* `profile.yaml`: Workload profile in GuideLLM format
* `traces.json`: vLLM server-side traces

**For this scenario, you only have benchmark data for 1 model (meta-llama/Llama-3.3-70B-Instruct) for varying TPs (2,4,8).**

For test, you should have similar benchmark data saved under the folder `test/`.

## Build BLIS

```
git clone -b unified_alphas git@github.com:inference-sim/inference-sim.git
go build -o simulation_worker main.go
```

## Train BLIS

To train both BLIS alpha and beta models for all benchmarks you have saved under `train/`:

```
python train_blis.py --train_results_path train/ --model_path models/
```

This gives you BLIS alpha and beta coefficients saved under `models/` to be used for testing/inference.

## Test BLIS

```
python test_blis.py --test_results_path test/ --model_path models/ --groupby_field tp/chunk_size/app/rps/model_path
```

Under `test_plots/blis`, you will find barplots showing average errors grouped by the field specified in the args.

## Bonus: Train QM

You can also train the [queueing model from llm-inferno](https://github.com/llm-inferno/model-trainer) using similar scripts as follows: 

```
python train_qm.py --train_results_path train/
```

This gives you files (one `QM_train.json` per TP folder under `train/`).

## Bonus: Test QM

```
python test_qm.py --test_results_path test/
```

This gives you a combined file `QM_test_TP=2/4/8.json` to feed into the Go binary.

# Scenario5++:

Same as Scenario5, with additional file `vllm.log` present in `train/` and `test/` folders. `total-kv-blocks` in `exp-config.yaml` is populated directly by parsing `vllm.log`. No support (yet) for Queueing model training/testing, but the same scripts should ideally work (untested).