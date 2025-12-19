# How to Run

## Train BLIS coefficients

There are two ways you can train BLIS:

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

> **Note:** If you rerun the exact same (LLM, TP, GPU, vllm-version) multiple times, the script simply overwrites existing coefficients in coeffs-filepath.

### FAQs:

* How do I change the cost function for the optimizer?

For this, go to the `train_blis.py` script and modify the constant `METRICS_IN_LOSS` list to include metrics in the cost function. Note that these metric names must match the column names in the GuideLLM Excel file.

* How do I run optimization over more/less iterations?

Similar to above, go to `train_blis.py` and update the constant `NUM_TPE_ITERS` to the number of iterations you want. In addition, you can now set the environment variable `NUM_TPE_ITERS` to desired iteration count before running the training as follows:

```
export NUM_TPE_ITERS=50
```

## Test

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

The test error plots will be saved to a folder `test_plots` in your current working directory. Depending on what arguments you provided for groupby, the plots will be saved as:

```
LLM=* TP=* GPU=* vllmVersion=*_error.png
LLM=granite-3.1-8b-instruct TP=* GPU=* vllmVersion=*_error.png
```

## Generate BLIS-simulated JSON file for [Compass](https://github.com/redhat-et/compass/blob/main/data/benchmarks.json):

To generate the final formatted synthetic metrics file, we currently take all the (LLM, TP, GPU, vllm-version) in the "Test" rows in the GuideLLM RH `.xlsx` file. We only take those (LLM, TP, GPU, vllm-version) combinations which have pretrained coefficients saved in the coeffs file. You can pass the config file and GuideLLM data filepath of your choice to the script. You can also pass the filename of the final synthetic metrics file to generate, as follows:

```
python generate_inference_json.py --coeffs-filepath coefficients.yaml --testing-filepath blis_rh_final.xlsx --synthetic-results-filepath benchmarks_BLIS.json --specs-filepath training_specs.csv
```

> **Note**: If you already have Compass data for a combination in `benchmarks_BLIS.json` and repeat the same combination in `training_specs.csv`, it will append the same results again to `benchmarks_BLIS.json`. To avoid this, **please ensure you do not repeat specs that you already have data for in `benchmarks_BLIS.json` in `training_specs.csv`.**

## Other util scripts

In this folder, you will also find two util scripts - `generate_training_specs.py` and `process_coeffs_yaml.py`. Here are their use-cases

* `generate_training_specs.py`: This script is used to extract all possible (LLM,TP,GPU,vllm-version) combinations from `blis_rh_final.xlsx` such that for each combination number of training rows >=4 and number of test rows >=2. Run this script to get an exhaustive list of all combinations to train BLIS over from RH GuideLLM data and save the combinations into `training_specs.csv`. 

```
python generate_training_specs.py
```

Beware: this will overwrite your existing `training_specs.csv`.

* `process_coeffs_yaml.py`: BLIS, when run only with `--model` expects default (TP,GPU,vllm-version) values. Currently the defaults are set by the logic: for each LLM, what is the (TP,GPU,vllm-version) combination that is most frequent in `blis_rh_final.xlsx`. Running this script modifies `coefficients.yaml` and populates these default values in the field `defaults` for all models present in `training_specs.csv`.

```
python process_coeffs_yaml.py
```
