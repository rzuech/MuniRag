import requests
import json
from config import LLM_MODEL, OLLAMA_HOST

def stream_answer(prompt):
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": True
    }

    with requests.post(url, json=payload, stream=True) as response:
        for line in response.iter_lines():
            if line:
                data_json = json.loads(line.decode('utf-8'))
                yield data_json.get('response', '')

# Old Code - Before 06/29/2025
#import requests
#from config import LLM_MODEL, OLLAMA_HOST

#def stream_answer(prompt):
#    url = f"{OLLAMA_HOST}/api/generate"
#    payload = {
#        "model": LLM_MODEL,
#        "prompt": prompt,
#        "stream": True
#    }
#
#    with requests.post(url, json=payload, stream=True) as response:
#        response.raise_for_status()
#        for line in response.iter_lines():
#            if line:
#                chunk = line.decode('utf-8')
#                if chunk.startswith('data: '):
#                    data_json = chunk[6:]
#                    if data_json.strip() == '[DONE]':
#                        break
#                    try:
#                        yield requests.utils.json.loads(data_json)['response']
#                    except:
#                        continue
#
#OLD Code - Before 06/27/2025
#import ollama, requests, os, json, time
#from config import LLM_MODEL, OLLAMA_HOST

#ollama.set_base_url(OLLAMA_HOST)

#def stream_answer(prompt):
#    for chunk in ollama.generate(LLM_MODEL, prompt=prompt, stream=True):
#        yield chunk["response"]
