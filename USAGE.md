## Step 1: Generate Response

write a configuration file to do inference, now supports 3 different frameworks: 'huggingface transformers', 'VLLM' and 'openai'.

The output file will be a jsonl file with `response` and `input_md5` columns.

### Huggingface Transformers

```json
{
    "model": {
        "model_name_or_path": "/apdcephfs_cq10/share_1324356/andyfei/data/Qwen2.5-1.5B-Instruct",
        "framework": "hf"
    },
    "output_file": "output/hf.json",
    "batch_size": 1
}
```

### VLLM

```json
{
    "model": {
        "model_name_or_path": "/apdcephfs_cq10/share_1324356/andyfei/data/Qwen2.5-1.5B-Instruct",
        "framework": "vllm"
    },
    "output_file": "output/vllm.json",
    "batch_size": 2
}
```

### openai API endpoint

```json
{
    "model": {
        "model_name_or_path": "model",
        "framework": "openai",
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "token-abc"
    },
    "output_file": "output/openai.json",
    "batch_size": 1
}
```

run generation using `python3 -m penguinscrolls.generate config.json`.

## Step 2: Evaluation using GPT-4o api

```bash
export OPENAI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# INPUT_FILE generated from step 1
python3 -m penguinscrolls.evaluate INPUT_FILE eval_result_dir/OUTPUT_FILE --concurrency 1
```

## Step 3: Collect and compare results

check [notebook](./notebook/collect_eval_result.ipynb)