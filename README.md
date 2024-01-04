# TGI Client

This is a utility to make it easier to run batched inference with a TGI server

## Usage

Run `python ./tgi-client/runner.py --input inputs.jsonl --output outputs.jsonl --model <HF Model ID> --endpoint http://127.0.0.1:8080` to batch process the prompts in `inputs.jsonl`. Prompts should not include the model preamble/template. Each line should be a dict like `{"prompt": "What is the meaning of life?"}`. The runner will ping the TGI instance until it's awake, then start working through the prompts.