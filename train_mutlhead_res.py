import torch
import torch.nn.functional as F
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig

from matplotlib import pyplot as plt
import copy
import datetime
import pandas as pd
from sklearn.metrics import accuracy_score
import time
import sys
import os

#sys.path.append('/home/xijian/pycharm_projects/document-level-classification/')
from config import *
from pre_data import load_data
from sklearn.model_selection import StratifiedShuffleSplit

tokenizer = XLNetTokenizer.from_pretrained('XLNet_pretrained_model/spiece.model')

ngpu = 1

use_cuda = torch.cuda.is_available() # 检测是否有可用的gpu
device = torch.device("cuda:0" if (use_cuda and ngpu>0) else "cpu")
print('*'*8, 'device:', device)



def split_dataset(csv_file, tokenizer, val_size=0.2, random_state=42):
    # 读取数据集
    df = pd.read_csv(csv_file)
    
    # 获取标签列
    labels = df['label'].values
    
    # 使用 StratifiedShuffleSplit 进行分层抽样
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(df, labels))
    
    # 分割数据集
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    # 打印每个集合中各类别的分布情况
    print("\nClass distribution in training set:")
    print(train_df['label'].value_counts(normalize=True))
    print("\nClass distribution in validation set:")
    print(val_df['label'].value_counts(normalize=True))
    
    # 保存分割后的数据集
    train_df.to_csv('train_data.csv', index=False)
    val_df.to_csv('val_data.csv', index=False)
    
    # 使用提供的load_data函数创建数据加载器
    train_dloader = load_data('train_data.csv', tokenizer, shuffle=True)
    val_dloader = load_data('val_data.csv', tokenizer)
    
    return train_dloader, val_dloader


    

loss_func = torch.nn.CrossEntropyLoss()
metric_func = lambda y_pred, y_true: accuracy_score(y_true, y_pred)
metric_name = 'acc'
df_history = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_"+metric_name])


# 打印时间
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m_%d %H:%M:%S')
    print('\n' + "=========="*8 + '%s'%nowtime)
                    
class MultiheadAttentionLayer(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiheadAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.head_dim = hidden_size // num_heads

        self.query = torch.nn.Linear(hidden_size, hidden_size)
        self.key = torch.nn.Linear(hidden_size, hidden_size)
        self.value = torch.nn.Linear(hidden_size, hidden_size)

        self.out_proj = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs):
        # inputs: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, hidden_size = inputs.size()

        # Linear transformations and split into heads
        queries = self.query(inputs).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key(inputs).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value(inputs).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        attention = torch.matmul(attn_weights, values)  # [batch_size, num_heads, seq_len, head_dim]

        # Concatenate heads and project back to hidden_size
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        out = self.out_proj(attention)  # [batch_size, seq_len, hidden_size]

        return out


class MyXLNetModel(torch.nn.Module):
    def __init__(self, pretrained_model_dir, num_classes, segment_len=150, dropout_p=0.5, num_heads=8):
        super(MyXLNetModel, self).__init__()

        self.seg_len = segment_len

        self.config = XLNetConfig.from_json_file('XLNet_pretrained_model/config.json')
        self.config.mem_len = 150  # enable the memory
        self.xlnet = XLNetModel.from_pretrained('XLNet_pretrained_model', config=self.config)

        if feature_extract:
            for p in self.xlnet.parameters():  # Freeze XLNet for feature extraction
                p.requires_grad = False

        d_model = self.config.hidden_size  # 768
        self.attention_layer1 = MultiheadAttentionLayer(d_model, num_heads)
        self.attention_layer2 = MultiheadAttentionLayer(d_model, num_heads)

        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.fc = torch.nn.Linear(d_model, num_classes)

    def get_segments_from_one_batch(self, input_ids, token_type_ids, attention_mask):
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)

        doc_len = input_ids.shape[1]
        q, r = divmod(doc_len, self.seg_len)
        split_chunks = [self.seg_len] * q + ([r] if r > 0 else [])

        input_ids = torch.split(input_ids, split_size_or_sections=split_chunks, dim=1)
        token_type_ids = torch.split(token_type_ids, split_size_or_sections=split_chunks, dim=1)
        attention_mask = torch.split(attention_mask, split_size_or_sections=split_chunks, dim=1)

        split_inputs = [{'input_ids': input_ids[seg_i], 'token_type_ids': token_type_ids[seg_i],
                         'attention_mask': attention_mask[seg_i]}
                        for seg_i in range(len(input_ids))]
        return split_inputs

    def forward(self, input_ids, token_type_ids, attention_mask):
        split_inputs = self.get_segments_from_one_batch(input_ids, token_type_ids, attention_mask)

        lower_intra_seg_repr = []
        mems = None
        for idx, seg_inp in enumerate(split_inputs):
            outputs = self.xlnet(**seg_inp, mems=mems)
            last_hidden = outputs.last_hidden_state
            mems = outputs.mems
            lower_intra_seg_repr.append(self.attention_layer1(last_hidden).mean(dim=1))

        lower_intra_seg_repr = torch.stack(lower_intra_seg_repr, dim=1)  # [batch_size, num_seg, hidden_size]
        higher_inter_seg_repr = self.attention_layer2(lower_intra_seg_repr).mean(dim=1)  # [batch_size, hidden_size]

        logits = self.fc(self.dropout(higher_inter_seg_repr))  # [batch_size, num_classes]
        return logits



def train_step(model, inps, labs, optimizer):
    input_ids, token_type_ids, attention_mask = inps
    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)
    labs = labs.to(device)

    model.train()  # 设置train mode
    optimizer.zero_grad()  # 梯度清零

    # forward
    logits = model(input_ids, token_type_ids, attention_mask)
    loss = loss_func(logits, labs)

    pred = torch.argmax(logits, dim=-1)
    metric = metric_func(pred.cpu().numpy(), labs.cpu().numpy()) # 返回的是tensor还是标量？

    # backward
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新参数

    return loss.item(), metric


def validate_step(model, inps, labs):
    input_ids, token_type_ids, attention_mask = inps
    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)
    labs = labs.to(device)

    model.eval()  # 设置eval mode

    # forward
    with torch.no_grad():
        logits = model(input_ids, token_type_ids, attention_mask)
        loss = loss_func(logits, labs)

        pred = torch.argmax(logits, dim=-1)
        metric = metric_func(pred.cpu().numpy(), labs.cpu().numpy())  # 返回的是tensor还是标量？

    return loss.item(), metric


def train_model(model, train_dloader, val_dloader, optimizer, scheduler_1r=None, init_epoch=0, num_epochs=10, print_every=150):
    starttime = time.time()
    print('*' * 27, 'start training...')
    printbar()

    best_metric = 0.
    for epoch in range(init_epoch+1, init_epoch+num_epochs+1):
        # 训练
        loss_sum, metric_sum = 0., 0.
        for step, (inp_ids, type_ids, att_mask, labs) in enumerate(train_dloader, start=1):
            inps = (inp_ids, type_ids, att_mask)
            loss, metric = train_step(model, inps, labs, optimizer)
            loss_sum += loss
            metric_sum += metric

            # 打印batch级别日志
            if step % print_every == 0:
                print('*'*27, f'[step = {step}] loss: {loss_sum/step:.3f}, {metric_name}: {metric_sum/step:.3f}')

        # 验证 一个epoch的train结束，做一次验证
        val_loss_sum, val_metric_sum = 0., 0.
        for val_step, (inp_ids, type_ids, att_mask, labs) in enumerate(val_dloader, start=1):
            inps = (inp_ids, type_ids, att_mask)
            val_loss, val_metric = validate_step(model, inps, labs)
            val_loss_sum += val_loss
            val_metric_sum += val_metric

        if scheduler_1r:
            scheduler_1r.step()

        # 记录和收集 1个epoch的训练和验证信息
        # columns=['epoch', 'loss', metric_name, 'val_loss', 'val_'+metric_name]
        record = (epoch, loss_sum/step, metric_sum/step, val_loss_sum/val_step, val_metric_sum/val_step)
        df_history.loc[epoch - 1] = record

        # 打印epoch级别日志
        print('EPOCH = {} loss: {:.3f}, {}: {:.3f}, val_loss: {:.3f}, val_{}: {:.3f}'.format(
               record[0], record[1], metric_name, record[2], record[3], metric_name, record[4]))
        printbar()

        # 保存最佳模型参数
        current_metric_avg = val_metric_sum/val_step
        if current_metric_avg > best_metric:
            best_metric = current_metric_avg
            checkpoint = save_dir + f'epoch{epoch:03d}_valacc{current_metric_avg:.3f}_ckpt.tar'
            if device.type == 'cuda' and ngpu > 1:
                model_sd = copy.deepcopy(model.module.state_dict())
            else:
                model_sd = copy.deepcopy(model.state_dict())
            # 保存
            torch.save({
                'loss': loss_sum / step,
                'epoch': epoch,
                'net': model_sd,
                'opt': optimizer.state_dict(),
            }, checkpoint)


    endtime = time.time()
    time_elapsed = endtime - starttime
    print('*' * 27, 'training finished...')
    print('*' * 27, 'and it costs {} h {} min {:.2f} s'.format(int(time_elapsed // 3600),
                                                               int((time_elapsed % 3600) // 60),
                                                               (time_elapsed % 3600) % 60))

    print('Best val Acc: {:4f}'.format(best_metric))
    return df_history


# 绘制训练曲线
def plot_metric(df_history, metric):
    plt.figure()

    train_metrics = df_history[metric]
    val_metrics = df_history['val_' + metric]  #

    epochs = range(1, len(train_metrics) + 1)

    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')  #

    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])

    plt.savefig(imgs_dir + 'xlnet_'+ metric + '.png')  # 保存图片
    plt.show()



if __name__ == '__main__':
    train_dloader, val_dloader = split_dataset('XlNet_merged_output.csv', tokenizer)

    sample_batch = next(iter(train_dloader))
    print('sample_batch:', len(sample_batch), sample_batch[0].size(), sample_batch[1].size(), sample_batch[2].size(),
          sample_batch[0].dtype, sample_batch[3].size(), sample_batch[3].dtype)  # 4   [b, doc_maxlen] int64


    model = MyXLNetModel(xlnet_model_dir, num_classes, segment_len=segment_len)
    model = model.to(device)
    if ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(ngpu)))  # 设置并行执行  device_ids=[0,1]

    init_epoch = 0
    # ===================================================================================================== new add
    # 不从头开始训练，而是从最新的那个checkpoint开始训练，或者可以收到指定从某个checkpoint开始训练
    if train_from_scrach is False and len(os.listdir(os.getcwd() + '/' + save_dir)) > 0:
        print('*' * 27, 'Loading model weights...')
        ckpt = torch.load(save_dir + last_new_checkpoint)  # dict  save在GPU 加载到 GPU
        init_epoch = int(last_new_checkpoint.split('_')[0][-3:])
        print('*' * 27, 'init_epoch=', init_epoch)
        model_sd = ckpt['net']
        if device.type == 'cuda' and ngpu > 1:
            model.module.load_state_dict(model_sd)
        else:
            model.load_state_dict(model_sd)
        print('*' * 27, 'Model loaded success!')
    # =====================================================================================================

    model.eval()
    sample_out = model(sample_batch[0], sample_batch[1], sample_batch[2])
    print('*' * 10, 'sample_out:', sample_out.shape)  # [b, 10]

    # 设置优化器
    print('Params to learn:')
    if feature_extract:  # 特征提取
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                # print('\t', name)
    else:  # 微调
        params_to_update = model.parameters()
        # for name, param in model.named_parameters():
        #     if param.requires_grad == True:
        #          print('\t', name)

    optimizer = torch.optim.Adam(params_to_update, lr=LR, weight_decay=1e-4)
    scheduler_1r = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lr_lambda=lambda epoch: 0.1 if epoch > EPOCHS * 0.8 else 1)

    train_model(model, train_dloader, val_dloader, optimizer, scheduler_1r,
                init_epoch=init_epoch, num_epochs=EPOCHS, print_every=50)

    plot_metric(df_history, 'loss')
    plot_metric(df_history, metric_name)