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
    return [json.loads(line) for line in lines]


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


def prepare_sentences(df):
    nodes = {}
    for _, row in df.iterrows():
        for key in ['src_ip_port', 'dest_ip_port']:
            node_id = row[key]
            nodes.setdefault(node_id, []).extend(row['phrase'])
    return list(nodes.values())


def train_FastText(events):
    """train FastText
    """
    print('Start training FastText')
    phrases = prepare_sentences(events)

    model = FastText(min_count=2, vector_size=64, workers=30, alpha=0.01, window=3, negative=3)
    model.build_vocab(phrases)
    model.train(phrases, epochs=100, total_examples=model.corpus_count)
    model.save(FASTTEXT_PATH)
    print(f'train model: {FASTTEXT_PATH}')


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


def Featurize(df):
    print('Start featuring')

    nodes = {} # {id of actor and object: }
    neimap = {}
    for _, row in df.iterrows():
        actor_id, object_id = row['src_ip_port'], row["dest_ip_port"]

        nodes.setdefault(actor_id, []).extend(row['phrase'])
        nodes.setdefault(object_id, []).extend(row['phrase'])

        neimap.setdefault(actor_id, set()).add(object_id)
        neimap.setdefault(object_id, set()).add(actor_id)

    features = []
    node_map_idx = {} # {node_id: index in features}

    for node, phrases in nodes.items():
        if len(phrases) > 1:
            features.append(infer(phrases))
            node_map_idx[node] = len(features) - 1

    print('finish featuring')

    return features, node_map_idx


def train_VAE(train_X):
    num_epochs = 8

    data_loader = DataLoader(train_X, batch_size=128, shuffle=True, pin_memory=True)
    model = VAE()
    model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    loss_list = []
    min_epoch_loss = 999
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, (x) in enumerate(data_loader):
            x = x.to(device)
            x_recon, mu, logvar = model(x)
            loss = loss_function(x_recon, x, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
            optimizer.step()
            train_loss += loss.item()

        epoch_loss = train_loss / len(data_loader.dataset)
        loss_list.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}")

        # if epoch_loss < min_epoch_loss:
        #     torch.save(model, './dataset/OpTC/FLASH/VAE.model')
        #     min_epoch_loss = epoch_loss
    torch.save(model.state_dict(), VAE_PATH)
    print('Training finish. ')

    # # draw training loss
    # plt.plot(loss_list)
    # plt.title('Training Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.grid(True)
    # plt.show()


# def get_MSE(model, features):
#     mse_loss = []
#     for vec in features:
#         x = torch.FloatTensor(vec).to(device)
#         x_recon, mu, logvar = model(x)
#         loss = F.mse_loss(x_recon, x, reduction='sum').item()
#         mse_loss.append(loss)
    
#     print('finish mse')

#     return mse_loss

def get_MSE(model, features):
    # 检查输入类型并优化转换
    if isinstance(features, list) and all(isinstance(arr, np.ndarray) for arr in features):
        features = np.stack(features)  # 合并多个NumPy数组
    
    x = torch.from_numpy(np.asarray(features)).float().to(device)  # 统一转换 n
    
    with torch.no_grad():
        x_recon, mu, logvar = model(x)
        mse_loss = F.mse_loss(x_recon, x, reduction='none').sum(dim=1).cpu().numpy()
    
    return mse_loss.tolist()


def get_threshold(model, features):
    validate_mse = get_MSE(model, features)
    threshold = np.percentile(validate_mse, 99.99)
    print('90: ',np.percentile(validate_mse,90))
    print('80: ',np.percentile(validate_mse,80))
    print('70: ',np.percentile(validate_mse,70))
    print('60: ',np.percentile(validate_mse,60))
    print(f'threshold: {threshold}')
    return threshold


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='CDM Parser')
    parser.add_argument("--dataset", type=str, default="optc_day23-flow")
    parser.add_argument("--f", type=bool, default=False)
    parser.add_argument("--v", type=bool, default=False)
    args = parser.parse_args()
    dataset = args.dataset
    dataset_path = f'./dataset/{dataset}/'
    IF_TRAIN_FASTTEXT = args.f
    IF_TRAIN_VAE = args.v

    dataset_file_map = {
        'optc_day23-flow' : {
            'train': 'head100000_train.json',
            'test': 'head100000_test.json',
            'ecarbro': 'ecarbro_24red_0201.json'
        }
    }

    TRAIN_FILE = f"{dataset_path}{dataset_file_map[dataset]['train']}"
    TEST_FILE = f"{dataset_path}{dataset_file_map[dataset]['test']}"
    EACRBRO_FILE = f"{dataset_path}{dataset_file_map[dataset]['ecarbro']}"
    OUTPUT_FILE = f'{dataset_path}net_alarms.txt'
    FASTTEXT_PATH = './models/FastText.model'
    VAE_PATH = './models/VAE.model'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = PositionalEncoder(64)

    # load train data
    df = load_data(TRAIN_FILE)

    # train FastText
    if IF_TRAIN_FASTTEXT:
        train_FastText(df)
    w2vmodel = FastText.load(FASTTEXT_PATH)

    end_time = time.time()
    print(f'Finish FastText train: {end_time - start_time}')

    # process and get feature, time consuming
    features, node_map_idx = Featurize(df)

    end_time = time.time()
    print(f'Finish training graph construct: {end_time - start_time}')

    # train VAE
    if not IF_TRAIN_VAE:
        print("Train VAE from scratch")
        train_X = CustomDataset(features)
        train_VAE(train_X)

    # eval
    model = VAE()  # 先重新创建模型结构
    model.load_state_dict(torch.load(VAE_PATH, map_location=device))
    model.to(device)  # 确保模型在正确的设备上
    model.eval()

    threshold = get_threshold(model, features)
    # threshold = 48.202

    # host related entities
    ip_objectID_map = {}
    with open(EACRBRO_FILE, 'r') as file:
        for line in file:
            event = json.loads(line)
            src_ip_port = f"{event['properties']['src_ip']} {event['properties']['src_port']}"
            dst_ip_port= f"{event['properties']['dest_ip']} {event['properties']['dest_port']}"

            for ip_port in [src_ip_port, dst_ip_port]:
                if ip_port not in ip_objectID_map:
                    ip_objectID_map[ip_port] = {event['objectID']}
                else:
                    ip_objectID_map[ip_port].add(event['objectID'])

    print(f'IP-ObjectID map size: {len(ip_objectID_map)}')

    end_time = time.time()
    print(f'Finish train VAE: {end_time - start_time}')

    # load test data
    test_df = load_data(TEST_FILE)
    test_features, test_node_index = Featurize(test_df)

    print(f'Finish test data feature extraction: {len(test_node_index)}')

    end_time = time.time()
    print(f'Finish testing graph construction: {end_time - start_time}')

    # get related entities
    related_features = []
    related_nodes = []
    i = 0
    j = 0
    for node, idx in test_node_index.items():
        if i < 5:
            print(f"Node {i}: '{node}' -> index {idx}")
        i += 1
        # if node in ip_objectID_map:
        #     if j < 5:
        #         print(f"Node {j}: '{node}' is related to {len(ip_objectID_map[node])} entities")
        #     j += 1
        #     related_features.append(test_features[idx])
        #     related_nodes.append(node)
        
        # 转换格式：从 "IP:PORT" 转为 "IP PORT"
        if ':' in node:
            # 将冒号替换为空格，匹配ip_objectID_map的格式
            converted_node = node.replace(':', ' ')
            
            if converted_node in ip_objectID_map:
                if j < 5:
                    print(f"✅ Matched Node {j}: '{node}' -> '{converted_node}' is related to {len(ip_objectID_map[converted_node])} entities")
                j += 1
                related_features.append(test_features[idx])
                related_nodes.append(node)  # 保存原始格式用于后续处理
            elif i <= 5:  # 只显示前5个未匹配的
                print(f"❌ No match for: '{node}' -> '{converted_node}'")
            else:
                if i <= 5:
                    print(f"⚠️ Invalid format (no ':' found): '{node}'")

    print(f"Total related nodes: {len(related_nodes)}")

    # test
    test_mse = []
    if related_features:
        test_mse = get_MSE(model, related_features)
    anomalies = set()
    for id, mse in zip(related_nodes, test_mse):
        if mse > threshold:
            anomalies.add(id)
    print(f'anomalies: {len(anomalies)}, total enenties: {len(related_nodes)}')

    anomaly_ecar = set()
    for ip_port in anomalies:
        anomaly_ecar |= ip_objectID_map[ip_port]

    print(f'host related anomalies: {len(anomaly_ecar)}')
    with open(OUTPUT_FILE, 'w') as file:
        for id in anomaly_ecar:
            file.write(id)
            file.write('\n')

    end_time = time.time()
    print(f'Finish eval: {end_time - start_time}')
