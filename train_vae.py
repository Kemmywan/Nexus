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


def train_VAE(train_X, num_epochs):

    data_loader = DataLoader(train_X, batch_size=128, shuffle=True, pin_memory=True)
    model = VAE()
    model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
    TEST_FILE = f"{dataset_path}{dataset_file_map[dataset]['test']}"
    EACRBRO_FILE = f"{dataset_path}{dataset_file_map[dataset]['ecarbro']}"
    OUTPUT_FILE = f'{dataset_path}net_alarms.txt'
    FASTTEXT_PATH = './models/FastText.model'
    VAE_PATH = './models/VAE.model'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load train data
    with open(f"dataset/{dataset}/train_features.pkl", "rb") as f:
        features = pkl.load(f)

    w2vmodel = FastText.load(FASTTEXT_PATH)

    # train VAE
    train_X = CustomDataset(features)
    train_VAE(train_X, 10)

    model = VAE().to(device)
    model.load_state_dict(torch.load(VAE_PATH, map_location=device))
    model.eval()
    threshold = get_threshold(model, features)
    print(f"threshold {threshold}")

    end_time = time.time()
    print(f'Finish eval: {end_time - start_time}')
