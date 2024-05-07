import json
import regex
import os
import hashlib
import copy

from tqdm import tqdm
import ijson
import tiktoken

def long_align_process(item):
    index = item["messages"][0]["content"].rfind("\n\n")
    text= item["messages"][0]["content"][:index]
    return item['id'], copy.deepcopy([text])
            
def github_repo_process(item):
    raw_string = item["code"]
    cleaned_string = image_pattern.sub('', raw_string)
    cleaned_string = hash_pattern.sub('', cleaned_string)
    return item['path'], copy.deepcopy([cleaned_string])
    
def wiki_process(item):
    raw_dict = item["ref_dict_list"]
    for txt in raw_dict:
        txt = xml_pattern.sub('', txt)
    return item['title'], copy.deepcopy(raw_dict)

def arxiv_process(item):
    txt_list = []
    for i in item["bbl_dict_list"]:
        for k, v in i.items():
            txt_list.append(v)
    return item['id'], copy.deepcopy(txt_list)

def books3_process(item):
    return item['meta']['name'] + "-" + item['meta']['src'], copy.deepcopy([item['text']])

source_path = {
    "long_align": "/mnt/data/user/tc_agi/llm/index_datasets/long_align_10_100k",
    "github_repo": "/mnt/data/user/tc_agi/llm/index_datasets/github_repo_10_100k",
    "wiki_en": "/mnt/data/user/tc_agi/llm/index_datasets/wiki_en_10_100k",
    "wiki_zh": "/mnt/data/user/tc_agi/llm/index_datasets/wiki_zh_10_100k",
    'arxiv': '/mnt/data/user/tc_agi/llm/index_datasets/arxiv_10_100k',
    'books3': '/mnt/data/user/tc_agi/llm/index_datasets/books3_10_100k',
}

process_func = {
    "long_align": long_align_process,
    "github_repo": github_repo_process,
    "wiki_en": wiki_process,
    "wiki_zh": wiki_process,
    'arxiv': arxiv_process,
    'books3': books3_process,
}


if __name__ == "__main__":          
    image_pattern = regex.compile(r'"image/png": "(.*?)",\n')
    hash_pattern = regex.compile(r'"hash": "(.*?)"\n')
    xml_pattern = regex.compile(r'<(.*?)>')
    
    tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
    
    output_path = "./seed_data"
    
    
    for data_source, data_path in source_path.items():
        in_path = data_path
        in_path = "./clean_data/"    
        meta_data = {
            'language': 'en',
            'nlines': 0,
            'nbytes': 0,    
            'length_distribute':{
              "less_4k": 0,
                "4k_8k": 0,
                "8k_16k": 0,
                "16k_32k": 0,
                "32k_64k": 0,
                "64k_128k": 0,
                "128k_256k": 0,
                "more_256k": 0,
            },
            'hdfs_path': "/user/tc_agi/llm/index_datasets/",
            'avg_token_per_line': 0,
            'data_sample': [],
        }
        
        total_token = 0
        os.makedirs(os.path.join(output_path, data_source), exist_ok=True)
        with open(os.path.join(in_path, 'data.jsonl'), 'r') as fin,\
            open(os.path.join(output_path, data_source, 'data.jsonl'), 'w') as fout:
            items = ijson.items(fin, "")
            for index, item in enumerate(tqdm(items)):
                indentifier, processed_txt_list = process_func[data_source](item)
                hash_id = hashlib.md5(str(processed_txt_list).encode()).hexdigest()
                token_len = len(tokenizer.encode(str(processed_txt_list)))
                new_item = {
                    'source': data_source,
                    'id': indentifier,
                    'data_len': len(processed_txt_list),
                    "length": len(str(processed_txt_list)),
                    "token_len": token_len,
                    'hash_id': hash_id,
                    "data": processed_txt_list,
                }
                
                if len(meta_data['data_sample']) < 10:
                    meta_data['data_sample'].append(new_item)
                
                meta_data['nlines'] += 1
                total_token += token_len
                
                if token_len < 4096:
                    meta_data['length_distribute']['less_4k'] += 1
                elif token_len < 8192:
                    meta_data['length_distribute']['4k_8k'] += 1
                elif token_len < 16384:
                    meta_data['length_distribute']['8k_16k'] += 1
                elif token_len < 32768:
                    meta_data['length_distribute']['16k_32k'] += 1
                elif token_len < 65536:
                    meta_data['length_distribute']['32k_64k'] += 1
                elif token_len < 131072:
                    meta_data['length_distribute']['64k_128k'] += 1
                elif token_len < 262144:
                    meta_data['length_distribute']['128k_256k'] += 1
                else:
                    meta_data['length_distribute']['more_256k'] += 1
                
                fout.write(json.dumps(new_item, ensure_ascii=False) + '\n')
        
        meta_data['nbytes'] = os.path.getsize(os.path.join('./seed_data', data_source, 'data.jsonl'))
        meta_data['avg_token_per_line'] = total_token / meta_data['nlines']
        
        with open(os.path.join(output_path, data_source, 'meta.json'), 'w') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=4)
        