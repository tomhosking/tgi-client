from args import parse_args
import jsonlines
from batched_tgi_job import TGIBatch

from transformers import AutoTokenizer

def main():

    args = parse_args()

    # with jsonlines.open(f'./inputs.jsonl') as reader:
    #     prompts = list(reader)

    prompts = [
        'Can you give me a recipe for a risotto?',
        'What is the meaning of life?'
    ]

    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    chat_turns = [[{"role": "user", "content": prompt}] for prompt in prompts]

    formatted_prompts = [tokenizer.apply_chat_template(chat, tokenize=False) for chat in chat_turns]



    job = TGIBatch(args.endpoint, formatted_prompts)

    job.try_connect()

    responses = job.run()

    with jsonlines.open(f'./output.jsonl','w') as writer:
        writer.write_all([{'response': resp.generated_text} for resp in responses])


if __name__ == "__main__":
    main()