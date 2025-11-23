import json
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from gensim.models import FastText
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
import argparse
from model import CustomDataset, VAE, loss_function
import pickle as pkl


np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)


class PositionalEncoder:
    """
    Position encoder similar to Transformer; copy from FLASH project.
    """
    def __init__(self, d_model, max_len=100000):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def embed(self, x):
        return x + self.pe[:x.size(0)]


def Sentence_Construction(entry):
    """优化字符串拼接方式（预分配内存）"""
    return f"{entry['src_ip_port']} {entry['dest_ip_port']} {entry['type']}".split()

def batch_json_parse(lines):
    """批量解析JSON（减少单行解析开销）"""
    # return [json.loads(line) for line in lines if line.strip()]
    result = []
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue  # skip empty lines
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"JSON decode error at batch line {idx}: {repr(line)}")
            print(f"Error: {e}")
            # Optionally, you can continue or raise depending on your need
            continue
    return result


def load_data(file_path, save_path=None, batch_size=10000, num_workers=4):
    print('Start loading')
    

    # 1. 批量读取文件（减少IO次数）
    with open(file_path, 'r') as f:
        lines = f.readlines()  # 全量读取（适合内存足够的情况）
    
    # 2. 分批次并行解析JSON
    batches = [lines[i:i + batch_size] for i in range(0, len(lines), batch_size)]
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        data_batches = list(executor.map(batch_json_parse, batches))
    
    # 3. 扁平化数据并处理
    data = list(chain.from_iterable(data_batches))
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for event in data:
            event['phrase'] = Sentence_Construction(event)
    
    # 4. 直接构造DataFrame（避免中间列表）
    df = pd.DataFrame(data)
    df.sort_values('timestamp', inplace=True)
    
    # 5. 使用更高效的存储格式（可选）
    if save_path:
        df.to_parquet(save_path)  # 比JSON快3-5倍
    
    print(f'Finish loading. Processed {len(df)} records')
    return df


def infer(document):
    word_embeddings = [w2vmodel.wv[word] for word in document if word in  w2vmodel.wv]
    
    if not word_embeddings:
        return np.zeros(64)

    combined_embeddings = np.array(word_embeddings)
    output_embedding = torch.tensor(combined_embeddings, dtype=torch.float)

    if len(document) < 100000:
        output_embedding = encoder.embed(output_embedding)

    output_embedding = output_embedding.detach().cpu().numpy()
    return np.mean(output_embedding, axis=0)


def construct_graph(df):
    print("Start construct graph. ")
    nodes = {} # {id of actor and object: }
    # neimap = {}
    for _, row in df.iterrows():
        actor_id, object_id = row['src_ip_port'], row["dest_ip_port"]

        nodes.setdefault(actor_id, []).extend(row['phrase'])
        nodes.setdefault(object_id, []).extend(row['phrase'])

        # neimap.setdefault(actor_id, set()).add(object_id)
        # neimap.setdefault(object_id, set()).add(actor_id)
    
    return nodes


def Featurize(nodes, dataset, type):
    print('Start featuring')

    features = []
    node_map_idx = {} # {node_id: index in features}

    for node, phrases in nodes.items():
        if len(phrases) > 1:
            features.append(infer(phrases))
            node_map_idx[node] = len(features) - 1

    print('finish featuring')

    with open(f"dataset/{dataset}/{type}_features.pkl", "wb") as f:
        pkl.dump(features, f)
    with open(f"dataset/{dataset}/{type}_node_map_idx.pkl", "wb") as f:
        pkl.dump(node_map_idx, f)

    return features, node_map_idx


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='CDM Parser')
    parser.add_argument("--dataset", type=str, default="optc_day23-flow")
    args = parser.parse_args()
    dataset = args.dataset
    dataset_path = f'./dataset/{dataset}/'
    dataset_file_map = {
        'optc_day23-flow' : {
            'train': 'train_conn_23_0-12.json',
            'test': 'test_conn_23_15-19.json',
            'ecarbro': 'ecarbro_23red_0201.json'
        },
        'optc_day24-flow' : {
            'train': 'train_conn_23_0-12.json',
            'test': 'test_conn_24_14-21.json',
            'ecarbro': 'ecarbro_24red_0501.json'
        },
        'optc_day25-flow' : {
            'train': 'train_conn_23_0-12.json',
            'test': 'test_conn_25_13-18.json',
            'ecarbro': 'ecarbro_25red_0051.json'
        }
    }

    TRAIN_FILE = './dataset/optc_day23-flow/train_conn_23_0-12.json'
    TEST_FILE = f"{dataset_path}{dataset_file_map[dataset]['test']}"
    EACRBRO_FILE = f"{dataset_path}{dataset_file_map[dataset]['ecarbro']}"
    OUTPUT_FILE = f'{dataset_path}net_alarms.txt'
    FASTTEXT_PATH = './models/FastText.model'
    VAE_PATH = './models/VAE.model'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = PositionalEncoder(64)


    # load train data, construct  graphs
    df = load_data(TRAIN_FILE)
    nodes = construct_graph(df)
    test_df = load_data(TEST_FILE)
    test_nodes = construct_graph(test_df)

    end_time = time.time()
    print(f"Mprof: end of constructing graphs {end_time - start_time}")

    # get feature encoding
    w2vmodel = FastText.load(FASTTEXT_PATH)
    features, node_map_idx = Featurize(nodes, dataset, 'train')
    test_features, test_node_index = Featurize(test_nodes, dataset, 'test')

    end_time = time.time()
    print(f'Finish eval: {end_time - start_time}')
