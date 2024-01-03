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

    def __init__(self, endpoint: str, prompts: list[str], silent=False, max_new_tokens=128, temperature=0.7) -> None:
        self.endpoint = endpoint
        self.prompts = prompts
        self.silent = silent
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    # TODO: This will currently eat ALL exceptions! Better to narrow it to Timeouts and "not ready" statuses
    @backoff.on_exception(backoff.constant, Exception, jitter=None, interval=30, max_time=60*15)
    def try_connect(self):
        print("Trying to connect...")
        resp = requests.post(
            self.endpoint + "/info",
            timeout=10,
        )
        payload = resp.json()
        if resp.status_code != 200:
            print(resp.status_code)
            raise Exception('Failed to connect to endpoint, status {:}'.format(resp.status_code))
        return payload
        
        


    def run(self):
        client = Client(self.endpoint, timeout=600)

        def generate_fn(text):
            return client.generate(text,max_new_tokens=self.max_new_tokens, temperature=self.temperature)

        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
            self.responses = list(tqdm(executor.map(generate_fn,self.prompts), total=len(self.prompts), disable=self.silent))

        return self.responses