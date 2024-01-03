# from tqdm import tqdm
import tqdm
from text_generation import AsyncClient, Client
# import asyncio
# from typing import Coroutine, List, Sequence
import jsonlines
import concurrent

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")


# encodeds = tokenizer.apply_chat_template(messages, tokenize=False)



def format_prompt_llama(instruction):
    template = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
<</SYS>>

{instruction} [/INST]"""
    return template.format(instruction=instruction)


def main():

    client = Client("http://127.0.0.1:80", timeout=600)

    def gen_text(text):
        return client.generate(text,max_new_tokens=128, temperature=0.7)


    for split in ['train','dev']:
        print('Running ', split)

        with jsonlines.open(f'./inputs.{split}.jsonl') as reader:
            prompts = list(reader)

        print('Loaded {:} inputs'.format(len(prompts)))
        ds = [format_prompt_llama(row['prompt']) for row in prompts]
        print('Formatted')


        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
            print('Thread pool started, running...')
            responses = list(tqdm.tqdm(executor.map(gen_text,ds), total=len(ds)))
            print('..done!')


        with jsonlines.open(f'./output.{split}.jsonl','w') as writer:
            writer.write_all([{'response': resp.generated_text} for resp in responses])


if __name__ == "__main__":
    print('Starting')
    
    main()

    print('Done!')