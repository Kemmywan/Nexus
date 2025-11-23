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

import torch.optim as optim

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


def load_data(file_path, save_path=None, batch_size=50000, num_workers=8):
    # print('Start loading')
    
    # # 预分配内存
    # all_data = []
 
    # # 1. 批量读取文件（减少IO次数）
    # with open(file_path, 'r') as f:
    #     lines = f.readlines()  # 全量读取（适合内存足够的情况）
    
    # # 2. 分批次并行解析JSON
    # batches = [lines[i:i + batch_size] for i in range(0, len(lines), batch_size)]

    # with ThreadPoolExecutor(max_workers=num_workers) as executor:
    #     data_batches = list(executor.map(batch_json_parse, batches))
    
    # # 3. 扁平化数据并处理
    # data = list(chain.from_iterable(data_batches))
    
    # # 并行构建phrase
    # def add_phrase(event):
    #     event['phrase'] = Sentence_Construction(event)
    #     return event
    
    # with ThreadPoolExecutor(max_workers=num_workers) as executor:
    #     data = list(executor.map(add_phrase, data))
    
    # # 4. 直接构造DataFrame（避免中间列表）
    # df = pd.DataFrame(data)
    # df.sort_values('timestamp', inplace=True)
        
    # print(f'Finish loading. Processed {len(df)} records')
    # return df


    # First Time

    # print('Start loading')
    # start_time = time.time()
    
    # # 预分配内存
    # all_data = []
 
    # # 1. 批量读取文件（减少IO次数）
    # print("📁 Step 1: Reading file...")
    # file_read_start = time.time()
    # with open(file_path, 'r') as f:
    #     lines = f.readlines()  # 全量读取（适合内存足够的情况）
    # file_read_time = time.time() - file_read_start
    # print(f"   ✅ File read complete: {len(lines):,} lines in {file_read_time:.2f}s ({len(lines)/file_read_time:.0f} lines/s)")
    
    # # 2. 分批次并行解析JSON
    # print("🔧 Step 2: Parsing JSON batches...")
    # json_parse_start = time.time()
    # batches = [lines[i:i + batch_size] for i in range(0, len(lines), batch_size)]
    # print(f"   📦 Created {len(batches)} batches of size {batch_size:,}")

    # with ThreadPoolExecutor(max_workers=num_workers) as executor:
    #     print(f"   🚀 Starting parallel JSON parsing with {num_workers} workers...")
    #     data_batches = []
    #     processed_batches = 0
        
    #     # Submit all tasks
    #     future_to_batch = {executor.submit(batch_json_parse, batch): i for i, batch in enumerate(batches)}
        
    #     # Process completed tasks
    #     for future in future_to_batch:
    #         batch_result = future.result()
    #         data_batches.append(batch_result)
    #         processed_batches += 1
            
    #         # Progress update every 10 batches or at the end
    #         if processed_batches % 10 == 0 or processed_batches == len(batches):
    #             elapsed = time.time() - json_parse_start
    #             progress = processed_batches / len(batches) * 100
    #             records_so_far = sum(len(batch) for batch in data_batches)
    #             speed = records_so_far / elapsed if elapsed > 0 else 0
    #             print(f"   📊 Progress: {processed_batches}/{len(batches)} batches ({progress:.1f}%) - {records_so_far:,} records in {elapsed:.1f}s ({speed:.0f} records/s)")
    
    # json_parse_time = time.time() - json_parse_start
    # print(f"   ✅ JSON parsing complete in {json_parse_time:.2f}s")
    
    # # 3. 扁平化数据并处理
    # print("🔗 Step 3: Flattening data...")
    # flatten_start = time.time()
    # data = list(chain.from_iterable(data_batches))
    # flatten_time = time.time() - flatten_start
    # print(f"   ✅ Flattening complete: {len(data):,} records in {flatten_time:.2f}s")
    
    # # 并行构建phrase
    # print("🔤 Step 4: Building phrases...")
    # phrase_start = time.time()
    
    # def add_phrase(event):
    #     event['phrase'] = Sentence_Construction(event)
    #     return event
    
    # print(f"   🚀 Starting parallel phrase construction with {num_workers} workers...")
    
    # # Process in chunks to show progress
    # chunk_size = 100000  # Process 100k records at a time for progress updates
    # processed_records = 0
    # processed_data = []
    
    # for i in range(0, len(data), chunk_size):
    #     chunk = data[i:i + chunk_size]
    #     chunk_start = time.time()
        
    #     with ThreadPoolExecutor(max_workers=num_workers) as executor:
    #         chunk_result = list(executor.map(add_phrase, chunk))
        
    #     processed_data.extend(chunk_result)
    #     processed_records += len(chunk)
        
    #     chunk_time = time.time() - chunk_start
    #     elapsed_total = time.time() - phrase_start
    #     progress = processed_records / len(data) * 100
    #     avg_speed = processed_records / elapsed_total if elapsed_total > 0 else 0
        
    #     print(f"   📊 Phrase progress: {processed_records:,}/{len(data):,} records ({progress:.1f}%) - Chunk: {chunk_time:.2f}s, Average: {avg_speed:.0f} records/s")
    
    # data = processed_data
    # phrase_time = time.time() - phrase_start
    # print(f"   ✅ Phrase construction complete in {phrase_time:.2f}s")
    
    # # 4. 直接构造DataFrame（避免中间列表）
    # print("📊 Step 5: Creating DataFrame...")
    # df_start = time.time()
    # df = pd.DataFrame(data)
    # df_creation_time = time.time() - df_start
    # print(f"   ✅ DataFrame created in {df_creation_time:.2f}s")
    
    # print("🔄 Step 6: Sorting by timestamp...")
    # sort_start = time.time()
    # df.sort_values('timestamp', inplace=True)
    # sort_time = time.time() - sort_start
    # print(f"   ✅ Sorting complete in {sort_time:.2f}s")
    
    # # 5. 使用更高效的存储格式（可选）
    # if save_path:
    #     print(f"💾 Step 7: Saving to {save_path}...")
    #     save_start = time.time()
    #     df.to_parquet(save_path)  # 比JSON快3-5倍
    #     save_time = time.time() - save_start
    #     print(f"   ✅ File saved in {save_time:.2f}s")
    
    # total_time = time.time() - start_time
    
    # # Final summary
    # print("\n" + "="*60)
    # print("📈 LOADING PERFORMANCE SUMMARY")
    # print("="*60)
    # print(f"Total records processed: {len(df):,}")
    # print(f"Total time: {total_time:.2f}s")
    # print(f"Overall speed: {len(df)/total_time:.0f} records/s")
    # print(f"Memory usage: ~{len(df) * 8 / 1024 / 1024:.1f} MB (estimated)")
    # print("\nBreakdown by step:")
    # print(f"  1. File reading:      {file_read_time:6.2f}s ({file_read_time/total_time*100:5.1f}%)")
    # print(f"  2. JSON parsing:      {json_parse_time:6.2f}s ({json_parse_time/total_time*100:5.1f}%)")
    # print(f"  3. Data flattening:   {flatten_time:6.2f}s ({flatten_time/total_time*100:5.1f}%)")
    # print(f"  4. Phrase building:   {phrase_time:6.2f}s ({phrase_time/total_time*100:5.1f}%)")
    # print(f"  5. DataFrame creation:{df_creation_time:6.2f}s ({df_creation_time/total_time*100:5.1f}%)")
    # print(f"  6. Sorting:           {sort_time:6.2f}s ({sort_time/total_time*100:5.1f}%)")
    # if save_path:
    #     print(f"  7. File saving:       {save_time:6.2f}s ({save_time/total_time*100:5.1f}%)")
    # print("="*60)
    
    # print(f'Finish loading. Processed {len(df)} records')
    # return df

    print('Start loading')
    start_time = time.time()
    
    # 1. 批量读取文件
    print("📁 Step 1: Reading file...")
    file_read_start = time.time()
    with open(file_path, 'r') as f:
        lines = f.readlines()
    file_read_time = time.time() - file_read_start
    print(f"   ✅ File read complete: {len(lines):,} lines in {file_read_time:.2f}s")
    
    # 2. 优化的JSON解析 + phrase构建（一步完成）
    print("🔧 Step 2: Parsing JSON + Building phrases...")
    parse_start = time.time()
    
    def optimized_batch_parse(lines):
        """一次性完成JSON解析和phrase构建"""
        result = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                # 直接构建phrase列表，避免字符串拼接和split
                event['phrase'] = [event['src_ip_port'], event['dest_ip_port'], event['type']]
                result.append(event)
            except (json.JSONDecodeError, KeyError):
                continue
        return result
    
    batches = [lines[i:i + batch_size] for i in range(0, len(lines), batch_size)]
    print(f"   📦 Created {len(batches)} batches, processing with {num_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        data_batches = []
        processed_batches = 0
        
        # 提交所有任务
        future_to_batch = {executor.submit(optimized_batch_parse, batch): i 
                          for i, batch in enumerate(batches)}
        
        # 处理完成的任务
        for future in future_to_batch:
            batch_result = future.result()
            data_batches.append(batch_result)
            processed_batches += 1
            
            # 每20个batch更新一次进度
            if processed_batches % 20 == 0 or processed_batches == len(batches):
                elapsed = time.time() - parse_start
                progress = processed_batches / len(batches) * 100
                records_so_far = sum(len(batch) for batch in data_batches)
                speed = records_so_far / elapsed if elapsed > 0 else 0
                print(f"   📊 Progress: {processed_batches}/{len(batches)} batches ({progress:.1f}%) - {speed:.0f} records/s")
    
    parse_time = time.time() - parse_start
    print(f"   ✅ JSON parsing + phrase building complete in {parse_time:.2f}s")
    
    # 3. 扁平化数据
    print("🔗 Step 3: Flattening data...")
    flatten_start = time.time()
    data = list(chain.from_iterable(data_batches))
    flatten_time = time.time() - flatten_start
    print(f"   ✅ Flattening complete: {len(data):,} records in {flatten_time:.2f}s")
    
    # 4. DataFrame创建
    print("📊 Step 4: Creating DataFrame...")
    df_start = time.time()
    df = pd.DataFrame(data)
    df_creation_time = time.time() - df_start
    print(f"   ✅ DataFrame created in {df_creation_time:.2f}s")
    
    # 5. 排序
    print("🔄 Step 5: Sorting by timestamp...")
    sort_start = time.time()
    df.sort_values('timestamp', inplace=True)
    sort_time = time.time() - sort_start
    print(f"   ✅ Sorting complete in {sort_time:.2f}s")
    
    # 6. 可选保存
    if save_path:
        print(f"💾 Step 6: Saving to {save_path}...")
        save_start = time.time()
        df.to_parquet(save_path)
        save_time = time.time() - save_start
        print(f"   ✅ File saved in {save_time:.2f}s")
    
    total_time = time.time() - start_time
    
    # 性能总结
    print("\n" + "="*60)
    print("📈 OPTIMIZED LOADING PERFORMANCE")
    print("="*60)
    print(f"Total records: {len(df):,}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Speed: {len(df)/total_time:.0f} records/s")
    print("\nTime breakdown:")
    print(f"  1. File reading:       {file_read_time:6.2f}s ({file_read_time/total_time*100:5.1f}%)")
    print(f"  2. Parse + Phrases:    {parse_time:6.2f}s ({parse_time/total_time*100:5.1f}%)")
    print(f"  3. Flattening:         {flatten_time:6.2f}s ({flatten_time/total_time*100:5.1f}%)")
    print(f"  4. DataFrame creation: {df_creation_time:6.2f}s ({df_creation_time/total_time*100:5.1f}%)")
    print(f"  5. Sorting:            {sort_time:6.2f}s ({sort_time/total_time*100:5.1f}%)")
    if save_path:
        print(f"  6. Saving:             {save_time:6.2f}s ({save_time/total_time*100:5.1f}%)")
    print("="*60)
    
    return df


def prepare_sentences(df):

    # nodes = {}
    # for _, row in df.iterrows():
    #     for key in ['src_ip_port', 'dest_ip_port']:
    #         node_id = row[key]
    #         nodes.setdefault(node_id, []).extend(row['phrase'])
    # return list(nodes.values())

    """优化的句子准备函数"""
    print("📝 Preparing sentences for FastText...")
    start_time = time.time()
    
    # 方法1: 向量化操作（最快）
    nodes = {}
    
    # 预提取列数据避免重复访问
    src_ports = df['src_ip_port'].values
    dest_ports = df['dest_ip_port'].values  
    phrases = df['phrase'].values
    
    for i in range(len(df)):
        src_id = src_ports[i]
        dest_id = dest_ports[i]
        phrase = phrases[i]
        
        # 使用extend而不是多次append
        if src_id not in nodes:
            nodes[src_id] = []
        if dest_id not in nodes:
            nodes[dest_id] = []
            
        nodes[src_id].extend(phrase)
        nodes[dest_id].extend(phrase)
    
    result = list(nodes.values())
    elapsed = time.time() - start_time
    print(f"   ✅ Sentence preparation complete: {len(result)} nodes in {elapsed:.2f}s")
    
    return result


def train_FastText(events):
    """train FastText
    """
    print('Start training FastText')
    phrases = prepare_sentences(events)

    print(f"Number of phrases: {len(phrases)}")

    model = FastText(min_count=2, vector_size=64, workers=16, alpha=0.01, window=3, negative=3)

    print(f"Number of phrases: {len(phrases)}")
    print(f"Sample phrase: {phrases[0]}")
    print(f"Phrase type: {type(phrases[0])}")

    model.build_vocab(phrases)

    print("Vocabulary size:", len(model.wv))

    model.train(phrases, epochs=10, total_examples=model.corpus_count)

    print("Training complete. Saving model...")

    model.save(FASTTEXT_PATH)

    print(f'train model: {FASTTEXT_PATH}')

def train_FastText_optimized(events):
    """优化的CPU FastText训练"""
    print('⚡ Start optimized CPU FastText training')
    start_time = time.time()
    
    phrases = prepare_sentences(events)
    print(f"Number of phrases: {len(phrases):,}")
    
    # 获取所有CPU核心
    import os
    max_workers = os.cpu_count()
    print(f"🖥️  Using all {max_workers} CPU cores")
    
    # 优化参数以最大化CPU性能
    model = FastText(
        min_count=5,           # 减少词汇量
        vector_size=64,        
        workers=max_workers,   # 使用所有CPU核心
        alpha=0.025,          # 提高学习率
        window=3,             
        negative=10,          # 增加negative sampling
        sample=1e-3,          # 启用subsampling
        sg=1,                # skip-gram
        hs=0,                # negative sampling
        epochs=5,            # 减少epochs
        max_vocab_size=30000, # 限制词汇量
        seed=42
    )
    
    print("📚 Building vocabulary...")
    vocab_start = time.time()
    model.build_vocab(phrases)
    vocab_time = time.time() - vocab_start
    print(f"   ✅ Vocabulary: {len(model.wv):,} words in {vocab_time:.2f}s")
    
    print("🚀 Training model...")
    train_start = time.time()
    model.train(phrases, total_examples=len(phrases), epochs=model.epochs)
    train_time = time.time() - train_start
    print(f"   ✅ Training complete in {train_time:.2f}s")
    
    model.save(FASTTEXT_PATH)
    
    total_time = time.time() - start_time
    print(f"⚡ Optimized CPU FastText finished: {total_time:.2f}s")
    
    return model


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
        }
    }

    TRAIN_FILE = f"{dataset_path}{dataset_file_map[dataset]['train']}"
    # TRAIN_FILE = f"./dataset/optc_day23-flow/head1000_train.json"
    TEST_FILE = f"{dataset_path}{dataset_file_map[dataset]['test']}"
    EACRBRO_FILE = f"{dataset_path}{dataset_file_map[dataset]['ecarbro']}"
    OUTPUT_FILE = f'{dataset_path}net_alarms.txt'
    FASTTEXT_PATH = './models/FastText.model'
    VAE_PATH = './models/VAE.model'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cuda'
    encoder = PositionalEncoder(64)

    # load train data
    df = load_data(TRAIN_FILE)
    # train FastText
    train_FastText_optimized(df)

    end_time = time.time()
    print(f'Finish train FastText: {end_time - start_time}')
