
#project_dir = '/home/xijian/pycharm_projects/document-level-classification/'
data_base_dir = 'XlNet_merged_output.csv'

xlnet_model_dir = 'XLNet_pretrained_model'

save_dir = 'train_model'
imgs_dir = 'img'

feature_extract = True # xlnet是否仅作为特征提取器，如果为否，则xlnet也参与训练，进行微调
train_from_scrach = True # 是否重头开始训练模型
last_new_checkpoint = 'epoch011_valacc0.971_ckpt.tar'

# 标签列表，针对二分类任务
labels = [0, 1]  # 0 和 1 分别代表两类

# 标签到索引的映射
label2id = {l: l for l in labels}  # 映射直接是自身
id2label = {l: l for l in labels}  # 映射直接是自身


LR = 5e-4 # 0.0005
EPOCHS = 20

doc_maxlen = 600 # 每个句子最大长度
segment_len = 150 # 段长
overlap = 50
num_classes = len(labels)

batch_size = 256