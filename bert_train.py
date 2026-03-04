# -*- coding: utf-8 -*-
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from datasets import load_dataset

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

# ======== 3. 实例化、编译 ============
bert_classifier = BertClassifier(bert_model)
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.BinaryCrossentropy()
metric = tf.keras.metrics.BinaryAccuracy(name='accuracy')

bert_classifier.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# ======== 4. 定义训练函数 ============
def train_classifier(sentences, labels, epochs=3):
    input_ids_list = []
    attention_masks_list = []
    
    for sentence in sentences:
        encoded_dict = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,  # 添加[CLS]和[SEP]
            max_length=64,            # 统一序列长度
            truncation=True,          # 显式截断过长文本（关键修复）
            padding='max_length',     # 替换pad_to_max_length（关键修复）
            return_attention_mask=True,
            return_tensors='tf'       # 返回TensorFlow格式
        )
        # 收集每个句子的编码结果
        input_ids_list.append(encoded_dict['input_ids'])
        attention_masks_list.append(encoded_dict['attention_mask'])
    
    # 拼接成批次数据（此时所有句子长度都是64，拼接不会报错）
    input_ids = tf.concat(input_ids_list, axis=0)
    attention_masks = tf.concat(attention_masks_list, axis=0)
    labels_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)  # 确保标签类型正确
    
    # 训练模型（输入是列表形式的两个张量）
    bert_classifier.fit(
        x=[input_ids, attention_masks],
        y=labels_tensor,
        epochs=epochs,
        batch_size=2  # 小批量适合示例数据
    )


# 训练模型并保存
if __name__ == "__main__":
 
    # 加载 IMDb 英文影评
    dataset = load_dataset("imdb")

    # 用更大的数据集训练
    real_sentences = [x['text'] for x in dataset['train']]
    real_labels = [1 if x['label'] == 1 else 0 for x in dataset['train']]
    
    train_classifier(real_sentences, real_labels, epochs=3)
    
    # 保存模型
    bert_classifier.save_weights('bert_sentiment_weights.weights.h5')