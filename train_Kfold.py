import torch
import torch.nn.functional as F
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig
from matplotlib import pyplot as plt
import copy
import datetime
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import time
import sys
import os
import numpy as np

from config import *
from pre_data_k import load_data

# 配置部分
BATCH_SIZE = 4
EPOCHS = 10
LR = 2e-5
segment_len = 150
num_classes = 10  # 根据实际分类数调整
feature_extract = True  # 是否只做特征提取
save_dir = 'checkpoints'
imgs_dir = 'imgs'
xlnet_model_dir = 'XLNet_pretrained_model/'

# 确保必要的目录存在
os.makedirs(save_dir, exist_ok=True)
os.makedirs(imgs_dir, exist_ok=True)

# 初始化tokenizer
tokenizer = XLNetTokenizer.from_pretrained('XLNet_pretrained_model/spiece.model')

# GPU设置
ngpu = 1
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if (use_cuda and ngpu > 0) else "cpu")
print('*'*8, 'device:', device)

# Focal Loss实现
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# 评估指标设置
loss_func = FocalLoss(alpha=1, gamma=2)
metric_func = lambda y_pred, y_true: accuracy_score(y_true, y_pred)
metric_name = 'acc'

# 时间打印函数
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m_%d %H:%M:%S')
    print('\n' + "=========="*8 + '%s'%nowtime)

# 注意力机制实现
class NyAttentioin(torch.nn.Module):
    def __init__(self, hidden_size, attensize_size):
        super(NyAttentioin, self).__init__()
        self.attn = torch.nn.Linear(hidden_size, attensize_size)
        self.ctx = torch.nn.Linear(attensize_size, 1, bias=False)

    def forward(self, inputs):
        u = self.attn(inputs).tanh()
        scores = self.ctx(u)
        attn_weights = F.softmax(scores, dim=1)
        out = torch.bmm(inputs.transpose(1, 2), attn_weights)
        return torch.squeeze(out, dim=-1)

# XLNet模型实现
class MyXLNetModel(torch.nn.Module):
    def __init__(self, pretrained_model_dir, num_classes, segment_len=150, dropout_p=0.5):
        super(MyXLNetModel, self).__init__()
        self.seg_len = segment_len
        
        self.config = XLNetConfig.from_json_file('XLNet_pretrained_model/config.json')
        self.config.mem_len = 150
        self.xlnet = XLNetModel.from_pretrained('XLNet_pretrained_model', config=self.config)
        
        if feature_extract:
            for p in self.xlnet.parameters():
                p.requires_grad = False
        
        d_model = self.config.hidden_size
        self.attention_layer1 = NyAttentioin(d_model, d_model // 2)
        self.attention_layer2 = NyAttentioin(d_model, d_model // 2)
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.fc = torch.nn.Linear(d_model, num_classes)
    
    def get_segments_from_one_batch(self, input_ids, token_type_ids, attention_mask):
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        doc_len = input_ids.shape[1]
        q, r = divmod(doc_len, self.seg_len)
        split_chunks = [self.seg_len] * q + [r] if r > 0 else [self.seg_len] * q
        
        input_ids = torch.split(input_ids, split_size_or_sections=split_chunks, dim=1)
        token_type_ids = torch.split(token_type_ids, split_size_or_sections=split_chunks, dim=1)
        attention_mask = torch.split(attention_mask, split_size_or_sections=split_chunks, dim=1)
        
        return [{'input_ids': ids, 'token_type_ids': tt_ids, 'attention_mask': att_mask} 
                for ids, tt_ids, att_mask in zip(input_ids, token_type_ids, attention_mask)]

    def forward(self, input_ids, token_type_ids, attention_mask):
        split_inputs = self.get_segments_from_one_batch(input_ids, token_type_ids, attention_mask)
        
        lower_intra_seg_repr = []
        mems = None
        for seg_inp in split_inputs:
            outputs = self.xlnet(**seg_inp, mems=mems)
            last_hidden = outputs.last_hidden_state
            mems = outputs.mems
            lower_intra_seg_repr.append(self.attention_layer1(last_hidden))
        
        lower_intra_seg_repr = torch.stack(lower_intra_seg_repr, dim=1)
        higher_inter_seg_repr = self.attention_layer2(lower_intra_seg_repr)
        
        return self.fc(self.dropout(higher_inter_seg_repr))

# 训练步骤函数
def train_step(model, inps, labs, optimizer):
    input_ids, token_type_ids, attention_mask = inps
    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)
    labs = labs.to(device)
    
    model.train()
    optimizer.zero_grad()
    
    logits = model(input_ids, token_type_ids, attention_mask)
    loss = loss_func(logits, labs)
    
    pred = torch.argmax(logits, dim=-1)
    metric = metric_func(pred.cpu().numpy(), labs.cpu().numpy())
    
    loss.backward()
    optimizer.step()
    
    return loss.item(), metric

# 验证步骤函数
def validate_step(model, inps, labs):
    input_ids, token_type_ids, attention_mask = inps
    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)
    labs = labs.to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, token_type_ids, attention_mask)
        loss = loss_func(logits, labs)
        pred = torch.argmax(logits, dim=-1)
        metric = metric_func(pred.cpu().numpy(), labs.cpu().numpy())
    
    return loss.item(), metric

def k_fold_train(model_class, full_dataset, k_folds=5, num_epochs=10, **model_params):
    """
    K折交叉验证训练函数
    参数：
        model_class: 模型类
        full_dataset: 完整数据集 (TensorDataset)
        k_folds: 折数
        num_epochs: 每折训练的轮数
        model_params: 模型参数
    """
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # 获取数据集的长度
    dataset_size = len(full_dataset)
    indices = np.arange(dataset_size)
    
    fold_results = []
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(indices)):
        print(f'开始第 {fold + 1} 折训练')
        printbar()
        
        # 创建训练集和验证集的采样器
        train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            full_dataset, 
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            num_workers=2
        )
        val_loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=BATCH_SIZE,
            sampler=val_sampler,
            num_workers=2
        )
        
        # 初始化模型
        model = model_class(**model_params)
        model = model.to(device)
        if ngpu > 1:
            model = torch.nn.DataParallel(model)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        
        # 训练当前折
        fold_history = train_fold(
            model=model,
            train_dloader=train_loader,
            val_dloader=val_loader,
            optimizer=optimizer,
            num_epochs=num_epochs,
            fold=fold
        )
        
        fold_results.append(fold_history)
        
    return fold_results

# 单折训练函数
def train_fold(model, train_dloader, val_dloader, optimizer, fold, num_epochs=10, print_every=150):
    starttime = time.time()
    print(f'开始第 {fold + 1} 折的训练...')
    printbar()
    
    best_metric = 0.
    fold_history = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_"+metric_name])
    
    for epoch in range(num_epochs):
        # 训练阶段
        loss_sum, metric_sum = 0., 0.
        for step, (inp_ids, type_ids, att_mask, labs) in enumerate(train_dloader, start=1):
            loss, metric = train_step(model, (inp_ids, type_ids, att_mask), labs, optimizer)
            loss_sum += loss
            metric_sum += metric
            
            if step % print_every == 0:
                print(f'第 {fold + 1} 折, Epoch {epoch + 1}, Step {step}: '
                      f'loss: {loss_sum/step:.3f}, {metric_name}: {metric_sum/step:.3f}')
        
        # 验证阶段
        val_loss_sum, val_metric_sum = 0., 0.
        for val_step, (inp_ids, type_ids, att_mask, labs) in enumerate(val_dloader, start=1):
            val_loss, val_metric = validate_step(model, (inp_ids, type_ids, att_mask), labs)
            val_loss_sum += val_loss
            val_metric_sum += val_metric
        
        # 记录结果
        record = (epoch + 1, loss_sum/step, metric_sum/step, val_loss_sum/val_step, val_metric_sum/val_step)
        fold_history.loc[epoch] = record
        
        # 保存最佳模型
        current_metric_avg = val_metric_sum/val_step
        if current_metric_avg > best_metric:
            best_metric = current_metric_avg
            checkpoint = save_dir + f'fold{fold + 1}_epoch{epoch + 1:03d}_valacc{current_metric_avg:.3f}_ckpt.tar'
            torch.save({
                'fold': fold + 1,
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_sum/step,
                'accuracy': current_metric_avg
            }, checkpoint)
        
        print(f'第 {fold + 1} 折, Epoch {epoch + 1}: '
              f'loss: {loss_sum/step:.3f}, {metric_name}: {metric_sum/step:.3f}, '
              f'val_loss: {val_loss_sum/val_step:.3f}, val_{metric_name}: {val_metric_sum/val_step:.3f}')
        printbar()
    
    return fold_history

# 绘制训练曲线
def plot_metric(df_history, metric, fold=None):
    plt.figure()
    
    train_metrics = df_history[metric]
    val_metrics = df_history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    
    title = f'Training and validation {metric}'
    if fold is not None:
        title += f' (Fold {fold + 1})'
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    
    save_path = imgs_dir + f'xlnet_{metric}_fold{fold + 1}.png' if fold is not None else imgs_dir + f'xlnet_{metric}.png'
    plt.savefig(save_path)
    plt.close()

# 主程序
if __name__ == '__main__':
    # 加载数据
    full_dataset = load_data('XlNet_merged_output.csv', tokenizer)
    
    # 模型参数
    model_params = {
        'pretrained_model_dir': xlnet_model_dir,
        'num_classes': num_classes,
        'segment_len': segment_len
    }
    
    # 执行K折交叉验证训练
    fold_results = k_fold_train(
        model_class=MyXLNetModel,
        full_dataset=full_dataset,
        k_folds=5,
        num_epochs=EPOCHS,
        **model_params
    )
    
    # 绘制结果
    for fold, history in enumerate(fold_results):
        plot_metric(history, 'loss', fold=fold)
        plot_metric(history, metric_name, fold=fold)
    
    # 计算并打印平均结果
    final_val_metrics = [df.iloc[-1]['val_' + metric_name] for df in fold_results]
    print(f'所有折的平均验证{metric_name}: {np.mean(final_val_metrics):.3f} ± {np.std(final_val_metrics):.3f}')