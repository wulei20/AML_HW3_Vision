import os
import json
POKEMON_FOLDER = 'pokemon-blip-captions'

file_names = os.listdir(POKEMON_FOLDER)

result_dict = {}

for item in file_names:
    name, ext = item.split('.')
    if name not in result_dict and ext in ('txt', 'jpg'):
        result_dict[name] = {}
    if ext == 'txt':
        with open(os.path.join(POKEMON_FOLDER, item), 'r') as f:
            result_dict[name]['text'] = f.read()
    elif ext == 'jpg':
        result_dict[name]['file_name'] = item
    else:
        print(f"Unknown item:{item}")

result_dict = {key : result_dict[key] for key in sorted(result_dict.keys())}

with open(os.path.join(POKEMON_FOLDER, 'metadata.jsonl'), 'w') as f:
    for item in result_dict:
        tmp = {key: result_dict[item][key] for key in sorted(result_dict[item].keys())}
        json.dump(tmp, f, indent=2)
        f.write('\n')