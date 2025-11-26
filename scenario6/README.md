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

### FAQs:

* How do I change the cost function for the optimizer?

For this, go to the `train_blis.py` script and modify the constant `METRICS_IN_LOSS` list to include metrics in the cost function. Note that these metric names must match the column names in the GuideLLM Excel file.

* How do I run optimization over more/less iterations?

Similar to above, go to `train_blis.py` and update the constant `NUM_TPE_ITERS` to the number of iterations you want.



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

## Generate synthetic JSON file similar to [Compass](https://github.com/redhat-et/compass/blob/main/data/benchmarks.json):

To generate the final formatted synthetic metrics file, we currently take all the "Test" rows in the GuideLLM RH `.xlsx` file. This means currently for each combination we report metrics for 50% of available data that we did not train over. We only take those (LLM, TP, GPU, vllm-version) combinations which have pretrained coefficients saved in the coeffs file. You can pass the config file and GuideLLM data filepath of your choice to the script. You can also pass the filename of the final synthetic metrics file to generate, as follows:

```
python generate_inference_json.py --coeffs-filepath coefficients.yaml --testing-filepath blis_rh_final.xlsx --synthetic-results-filepath benchmarks_BLIS.json
```