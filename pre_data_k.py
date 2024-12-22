import time
import pandas as pd
import torch
from transformers import XLNetTokenizer

import sys
#sys.path.append('/home/xijian/pycharm_projects/document-level-classification/')
from config import *

# 加载XLNet分词器
tokenizer = XLNetTokenizer.from_pretrained('XLNet_pretrained_model/spiece.model')

# 读取数据：content为文本列，分类标签为标签列
def read_data(filepath, tokenizer):
    df_data = pd.read_csv(filepath, encoding='UTF-8-sig')  # 默认从CSV文件加载数据
    df_data = df_data.dropna()

    # 获取文本和标签列
    x_data, y_data = df_data['content'], df_data['分类标签']
    print('*' * 27, f'Data Loaded: {x_data.shape[0]} samples')

    # 文本数据编码
    x_data = xlnet_encode(x_data, tokenizer)

    # 标签转为tensor格式
    y_data = torch.tensor(y_data.values, dtype=torch.long)

    return x_data, y_data

# 文本编码
def xlnet_encode(texts, tokenizer):
    starttime = time.time()
    print('*' * 27, 'Start encoding...')
    inputs = tokenizer.batch_encode_plus(
        texts.tolist(),  # 转为列表以适配 tokenizer
        return_tensors='pt',
        add_special_tokens=True,
        max_length=doc_maxlen,  # 最大长度限制
        padding='longest',  # 填充到最长序列
        truncation=True  # 超过最大长度时截断
    )
    endtime = time.time()
    print('*' * 27, 'Data encoded to IDs.')
    print('*' * 27, f'Encoding time: {int((endtime - starttime) // 60)} min {((endtime - starttime) % 60):.2f} s')
    return inputs

# 数据加载器
def load_data(filepath, tokenizer, shuffle=False):
    inputs, y_data = read_data(filepath, tokenizer)
    # 创建TensorDataset
    inp_dset = torch.utils.data.TensorDataset(
        inputs['input_ids'], 
        inputs['token_type_ids'], 
        inputs['attention_mask'], 
        y_data
    )
    # 直接返回TensorDataset，而不是DataLoader
    return inp_dset
if __name__ == '__main__':
    # 测试加载数据
    data_loader = load_data('XlNet_merged_output.csv', tokenizer)  # 替换为你的CSV文件路径
    x_0, x_1, x_2, y = next(iter(data_loader))
    print('Sample:', 
          'x_0 (input_ids):', x_0.shape, 
          'x_1 (token_type_ids):', x_1.shape, 
          'x_2 (attention_mask):', x_2.shape, 
          'y (labels):', y.shape)
