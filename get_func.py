import requests
import json
with open('C:/Users/HP/AppData/Local/BUTLER/config.json', encoding='utf-8') as f:
    data = json.load(f)
    api = data.get('nvidia_api_keys', [{'key':''}])
    api = api[0]['key'] if api else ''

url = 'https://api.nvcf.nvidia.com/v2/nvcf/functions'
res = requests.get(url, headers={'Authorization': 'Bearer ' + api})
if res.status_code == 200:
    for x in res.json().get('functions', []):
        if 'magpie' in x.get('name', '').lower() or 'speech' in x.get('name', '').lower() or 'tts' in x.get('name', '').lower():
            print(f"{x['name']}: {x['id']}")
else:
    print(f"Error {res.status_code}: {res.text}")
