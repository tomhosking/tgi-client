from text_generation import Client
import concurrent
from tqdm import tqdm
import backoff
import requests

class TGIBatch:

    endpoint: str
    prompts: list[str]
    responses: list[str]
    silent: bool
    max_new_tokens: int
    temperature: float
    bsz: int

    def __init__(self, endpoint: str, prompts: list[str], silent: bool=False, max_new_tokens: int=256, temperature: float=0.7, bsz: int = 16) -> None:
        self.endpoint = endpoint
        self.prompts = prompts
        self.silent = silent
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.bsz = bsz

    # TODO: This will currently eat ALL exceptions! Better to narrow it to Timeouts and "not ready" statuses
    @backoff.on_exception(backoff.constant, Exception, jitter=None, interval=30, max_time=60*15)
    def try_connect(self):
        print("Trying to connect...")
        resp = requests.get(
            self.endpoint + "/info",
            timeout=10,
        )
        payload = resp.json()
        if resp.status_code != 200:
            print(resp.status_code)
            raise Exception('Failed to connect to endpoint, status {:}'.format(resp.status_code))
        print("..server is up!")
        return payload
        
        


    def run(self):
        client = Client(self.endpoint, timeout=600)

        def generate_fn(example):
            # TODO: Capture response metadata/errors
            return {'response': client.generate(example['input'],max_new_tokens=self.max_new_tokens, temperature=self.temperature), **example}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.bsz) as executor:
            self.responses = list(tqdm(executor.map(generate_fn,self.prompts), total=len(self.prompts), disable=self.silent))

        return self.responses