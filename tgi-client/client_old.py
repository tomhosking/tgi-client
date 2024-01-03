from tqdm import tqdm
import tqdm
from text_generation import AsyncClient
import asyncio
from typing import Coroutine, List, Sequence
import jsonlines


def _limit_concurrency(
    coroutines: Sequence[Coroutine], concurrency: int
) -> List[Coroutine]:
    """Decorate coroutines to limit concurrency.
    Enforces a limit on the number of coroutines that can run concurrently in higher
    level asyncio-compatible concurrency managers like asyncio.gather(coroutines) and
    asyncio.as_completed(coroutines).
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def with_concurrency_limit(coroutine: Coroutine) -> Coroutine:
        async with semaphore:
            return await coroutine

    return [with_concurrency_limit(coroutine) for coroutine in coroutines]

def format_prompt_llama(instruction):
    template = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
<</SYS>>

{instruction} [/INST]"""
    return template.format(instruction=instruction)


def format_prompt_mistral(instruction):
    return """<s>[INST] {instruction} [/INST]""".format(instruction=instruction)

def get_response(client, prompt):
    return {
        'response': client.generate(prompt, max_new_tokens=100, temperature=0.7).generated_text,
        'prompt': prompt,
    }


async def main():

    client = AsyncClient("http://127.0.0.1:80", timeout=600)



    for split in ['train','dev']:
        print('Running ', split)

        with jsonlines.open(f'./inputs.{split}.jsonl') as reader:
            prompts = list(reader)

        print('Loaded {:} inputs'.format(len(prompts)))
        # ds = [format_prompt_llama(row['prompt']) for row in prompts]
        ds = [format_prompt_mistral(row['prompt']) for row in prompts]


        fs = [get_response(prompt) for prompt in ds]
        fs = _limit_concurrency(fs,128)


        responses = await tqdm.asyncio.tqdm_asyncio.gather(*fs, total=len(fs))


        with jsonlines.open(f'./output.{split}.jsonl','w') as writer:
            writer.write_all(responses)


if __name__ == "__main__":
    print('Starting')
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

    print('Done!')