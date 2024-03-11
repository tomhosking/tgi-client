from args import parse_args
import jsonlines
from batched_tgi_job import TGIBatch

from transformers import AutoTokenizer

def main():

    args = parse_args()

    with jsonlines.open(f'{args.input}') as reader:
        prompts = list(reader)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    chat_turns = [{'turns': [{"role": "user", "content": prompt['prompt']}], 'orig': prompt} for prompt in prompts]

    formatted_prompts = [{'input': tokenizer.apply_chat_template(chat['turns'], tokenize=False), 'orig': chat['orig']} for chat in chat_turns]



    job = TGIBatch(args.endpoint, formatted_prompts)

    job.try_connect()

    responses = job.run()

    with jsonlines.open(f'{args.output}','w') as writer:
        writer.write_all([{'response': resp['response'].generated_text, **resp['orig']} for resp in responses])


if __name__ == "__main__":
    main()