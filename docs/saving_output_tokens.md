# How to Save Output Tokens

To save the generated output tokens from a benchmark run, you need to make a small modification to the `benchmarks/benchmark_serving_simulator.py` script.

## Configuration

1.  **Open `benchmarks/benchmark_serving_simulator.py`**.
2.  **Enable Token Saving**: Locate the `WRITE_OUTPUT_TOKENS` global variable at the top of the file and set it to `True`.
    ```python
    WRITE_OUTPUT_TOKENS = True
    ```
3.  **Set Model Name**: Set the `MODEL_NAME` global variable to the name of the model you are benchmarking. This name is used for creating the output directory. **Ensure the model name does not contain slashes (`/`)**.
    ```python
    MODEL_NAME = "Qwen2-7B"  # Default model name, make sure no slashes are present
    ```

## Output Location and Format

-   **Output Directory**: The generated files will be saved in a directory named after the model inside `benchmarks/output_tokens/`. For example: `benchmarks/output_tokens/Qwen2-7B/`.
-   **File Naming Convention**: The output files are named using the format `output_tokens_{num_requests}.json`, where `{num_requests}` is the number of successful requests in the benchmark run.
-   **File Format**: The output file is a JSON file containing a list of conversation objects. Each object represents a request and its corresponding generated response.

### Example Output (`output_tokens_100.json`)

```json
[
  {
    "id": "request_0",
    "conversations": [
      {
        "from": "human",
        "value": "What is the capital of France?"
      },
      {
        "from": "gpt",
        "value": "The capital of France is Paris."
      }
    ]
  },
  {
    "id": "request_1",
    "conversations": [
      {
        "from": "human",
        "value": "Write a short poem about AI."
      },
      {
        "from": "gpt",
        "value": "In circuits of silicon, a mind takes flight,\nWith algorithms learning, day and night.\nA new consciousness, in code it's cast,\nThe future's dawn, arriving fast."
      }
    ]
  }
]
```
