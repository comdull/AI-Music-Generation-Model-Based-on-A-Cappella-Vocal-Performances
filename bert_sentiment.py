# -*- coding: utf-8 -*-
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model

# ======== 1. 初始化分词器和预训练BERT ============
# 加载分词器和模型（指定from_pt=True确保TensorFlow模型正确加载PyTorch权重）
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased', from_pt=True)

# ======== 2. 自定义分类器 ============
class BertClassifier(Model):
    def __init__(self, bert):
        super(BertClassifier, self).__init__()
        self.bert = bert
        self.dropout = Dropout(0.3)  # 显式导入Dropout，避免依赖全局
        self.classifier = Dense(1, activation='sigmoid')

    # 修正call方法，确保输入正确传递给BERT
    def call(self, inputs):
        # inputs是一个列表：[input_ids, attention_mask]
        input_ids, attention_mask = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS]位置的输出
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

# ======== 3. 实例化并加载预训练权重 ============
bert_classifier = BertClassifier(bert_model)

# 必须先“跑一遍”创建权重变量
dummy_inputs = [
    tf.constant([[1]*64]),
    tf.constant([[1]*64])
]
bert_classifier(dummy_inputs)

# 加载保存好的权重
bert_classifier.load_weights('bert_sentiment_weights.weights.h5')
print("BERT sentiment model weights loaded.")

# ======== 4. 提供外部可调用的情感分析函数 ============
def analyze_lyrics_sentiment(lyrics_text: str):
    encoded_dict = tokenizer.encode_plus(
        lyrics_text,
        add_special_tokens=True,
        max_length=64,
        truncation=True,          # 同样添加截断
        padding='max_length',     # 同样使用新参数
        return_attention_mask=True,
        return_tensors='tf'
    )
    input_id = encoded_dict['input_ids']
    attn_mask = encoded_dict['attention_mask']
    
    # 预测（修复张量索引，根据实际输出形状调整）
    prediction = bert_classifier.predict([input_id, attn_mask])[0][0]  # 正确的索引方式
    confidence = float(prediction)  # 直接转换为浮点数
    
    # 判断情感标签
    if confidence >= 0.5:
        label = "POSITIVE"
    else:
        label = "NEGATIVE"
    return label, round(confidence, 3)

# ======== 5. 示例使用 ============
if __name__ == "__main__":
    # 测试情感分析
    test_lyrics = "When I see your smile, I feel so good."
    label, conf = analyze_lyrics_sentiment(test_lyrics)
    print(f"Lyrics sentiment: {label} (confidence: {conf})")
    
    test_lyrics2 = "Lonely nights, empty rooms, everything is wrong.I hate you!"
    label2, conf2 = analyze_lyrics_sentiment(test_lyrics2)
    print(f"Lyrics sentiment: {label2} (confidence: {conf2})")