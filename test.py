import torch
import torch.nn.functional as F
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig
from config import *

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
        batch_size, seq_len, hidden_size = inputs.size()

        queries = self.query(inputs).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key(inputs).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value(inputs).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attention = torch.matmul(attn_weights, values)

        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        out = self.out_proj(attention)

        return out

class MyXLNetModel(torch.nn.Module):
    def __init__(self, pretrained_model_dir, num_classes, segment_len=150, dropout_p=0.5, num_heads=8, feature_extract=True):
        super(MyXLNetModel, self).__init__()

        self.seg_len = segment_len
        self.feature_extract = feature_extract

        self.config = XLNetConfig.from_pretrained(pretrained_model_dir)
        self.config.mem_len = segment_len
        self.xlnet = XLNetModel.from_pretrained(pretrained_model_dir, config=self.config)

        if self.feature_extract:
            for p in self.xlnet.parameters():
                p.requires_grad = False

        d_model = self.config.hidden_size
        self.attention_layer1 = MultiheadAttentionLayer(d_model, num_heads)
        self.attention_layer2 = MultiheadAttentionLayer(d_model, num_heads)

        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.fc = torch.nn.Linear(d_model, num_classes)

    def get_segments_from_one_batch(self, input_ids, token_type_ids, attention_mask):
        try:
            input_ids = input_ids.to(self.xlnet.device)
            token_type_ids = token_type_ids.to(self.xlnet.device)
            attention_mask = attention_mask.to(self.xlnet.device)

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
        except Exception as e:
            raise ValueError(f"Error splitting input batch: {e}")

    def forward(self, input_ids, token_type_ids, attention_mask):
        split_inputs = self.get_segments_from_one_batch(input_ids, token_type_ids, attention_mask)

        lower_intra_seg_repr = []
        mems = None
        for seg_inp in split_inputs:
            outputs = self.xlnet(**seg_inp, mems=mems)
            last_hidden = outputs.last_hidden_state
            mems = outputs.mems
            lower_intra_seg_repr.append(self.attention_layer1(last_hidden).mean(dim=1))

        lower_intra_seg_repr = torch.stack(lower_intra_seg_repr, dim=1)
        higher_inter_seg_repr = self.attention_layer2(lower_intra_seg_repr).mean(dim=1)

        logits = self.fc(self.dropout(higher_inter_seg_repr))
        return logits

def predict_sentiment(text, model, tokenizer, device, max_length=512):
    text = str(text)

    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    token_type_ids = inputs['token_type_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(input_ids, token_type_ids, attention_mask)
        predictions = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(predictions, dim=1).item()
        confidence = predictions[0][predicted_class].item()

    return predicted_class, confidence

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 2
    checkpoint_path = "train_modelepoch002_valacc0.928_ckpt.tar"
    tokenizer = XLNetTokenizer.from_pretrained('XLNet_pretrained_model', do_lower_case=True, keep_accents=True)
    model = MyXLNetModel('XLNet_pretrained_model', num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['net'])
    model = model.to(device)

    text = """
12月27日，随着滨州滨港华能集中式光伏项目建成投产，山东风电与光伏装机容量历史性突破1亿千瓦大关，达到1.002亿千瓦。山东成为全国首个风光装机过亿千瓦的省级电网。

“接得上”才能“送得出”

强大的电网是新能源发展的基础。山东已建成“五交三直一环网”全国最大的省域交直流混联电网，全国首条“风光火储一体化”送电的陇东-山东特高压线路工程也已实现全线贯通。坚强可靠的主网架有力保障了新能源“接得上”“送得出”。

以鲁北盐碱滩涂地为例，这里是山东省千万千瓦级的新能源建设基地，为了把这些绿电输送出去，国网山东电力完善北部500千伏电网架构，加快建设潍坊、滨州、东营区域500千伏新能源汇集工程，形成了北部可再生能源基地“北电南送”的输电通道，曾经的不毛之地如今已“绿”意盎然

坚强的主网架有力支撑了山东五大基地送出，而可靠的配网则支撑了分布式新能源的友好发展。山东是分布式光伏装机第一大省，为了有力支撑分布式新能源接入，国网山东电力科学规划布局配电网，10千伏线路联络率达到99.3%。同时，还积极创新探索分布式光伏“集中汇流、升压并网”建设模式，试点开展“云储能”建设，助力破解分布式光伏“成长的烦恼”。

如果说坚强电网是新能源发展的“硬件设施”，那么“清风暖阳”则是新能源发展的“软件服务”。

国网山东电力创新开展“清风暖阳”特色行动，打造了专业高效、开放友好的新能源并网服务，全省光伏、风电装机分别达7352.53万千瓦、2668.78万千瓦，较去年底增长29.16%、3.00%，其中光伏装机规模连续7年位居全国首位。

“消纳得了”才能“用得好”

风光等新能源不同于传统的火电、水电等可控性较强的电源，其发电具有随机性、波动性和不确定性特点，天气炎热、空调大开的时候往往没有风，傍晚用电高峰又没有光，这便是“极热无风”“晚峰无光”现象。

也正是由于新能源的特点，电有了更为突出的时空价值、绿色价值、安全价值。而如何体现好这些价值，电力市场在其中发挥了重要作用。今年6月17日，山东电力现货市场转入正式运行，依托成熟的现货市场，国网山东电力推动全省全部集中式风电光伏场站参与电力市场交易。今年以来达成绿电交易22.3亿千瓦时、同比增长37%，充分体现新能源绿色环境价值。

山东还在全国率先推出了分时电价动态调整机制、深谷电价机制，将现货市场分时价格信号全链条传导至终端用户。在春、秋季午间时段新能源大发，电力供过于求，市场价格偏低，激励用户尽可能地多用绿色电、低价电，更好地促进能源转型；在冬夏负荷晚高峰，光伏出力为零，电力供需偏紧，市场价格较高，引导用户错峰生产，更好地支撑电力保供。

而随着跨省跨区电力市场交易机制的不断完善，山东不断拓展更远距离的资源优化配置，组织开展省内富余新能源参与省间现货交易，以市场化手段实现新能源大范围消纳。"""
    
    sentiment_map = {0: "负面", 1: "正面"}
    print("\n文本情感分析结果：")
    print("-" * 50)
    
    try:
        class_id, confidence = predict_sentiment(text, model, tokenizer, device)
        sentiment = sentiment_map.get(class_id, "未知")
        print(f"情感倾向: {sentiment}")
        print(f"置信度: {confidence:.2%}")
        print("-" * 50)
    except Exception as e:
        print(f"处理文本时出错: {e}")
        print("-" * 50)