import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset, load_metric
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font='IPAGothic')

# wrime の読み込み
df_wrime = pd.read_table("/workspace/wrime/wrime-ver1.tsv")
df_wrime.head(2)

# 8感情のリスト
emotion_names = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']
emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']
num_labels = len(emotion_names)

df_wrime['readers_emotion_intensities'] = df_wrime.apply(lambda x: [x['Avg. Readers_' + name] for name in emotion_names], axis=1)

is_target = df_wrime['readers_emotion_intensities'].map(lambda x: max(x) >= 2)
df_wrime_target = df_wrime[is_target]

# train / testに分割
df_groups = df_wrime_target.groupby('Train/Dev/Test')
df_train = df_groups.get_group('train')
df_test = pd.concat([df_groups.get_group('dev'), df_groups.get_group('test')])
print('train :', len(df_train))
print('test :', len(df_test))

# モデル指定・Tokenizer読み込み
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
# model_name = "cl-tohoku/bert-base-japanese-v3"
# model_name = "cl-tohoku/bert-large-japanese-v2"
# model_name = "cl-tohoku/bert-base-japanese-char-v3"
# model_name = "cl-tohoku/bert-large-japanese-char-v2"
checkpoint = model_name
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 前処理・感情強度の正規化(総和=1)
def tokenize_function(batch):
	tokenized_batch = tokenizer(batch['Sentence'], truncation=True, padding="max_length")
	tokenized_batch['labels'] = [x / np.sum(x) for x in batch['readers_emotion_intensities']]
	return tokenized_batch

# transformers用のデータセット形式に変換
target_columns = ['Sentence', 'readers_emotion_intensities']
train_dataset = Dataset.from_pandas(df_train[target_columns])
test_dataset = Dataset.from_pandas(df_test[target_columns])

# 前処理の適用
train_tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
test_tokenized_dataset = test_dataset.map(tokenize_function, batched=True)

# 学習用モデル指定
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)

# 評価指標を定義
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
	logits, labels = eval_pred
	predictions = np.argmax(logits, axis=-1)
	label_ids = np.argmax(labels, axis=-1)
	return metric.compute(predictions=predictions, references=label_ids)

# 訓練時の設定(推論時はコメントアウト)
training_args = TrainingArguments(
	output_dir=model_name,
	per_device_train_batch_size=8,
	num_train_epochs=10,
	evaluation_strategy="epoch",
	load_best_model_at_end=True,
	save_strategy='epoch'
)

# trainerを生成
trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=train_tokenized_dataset,
	eval_dataset=test_tokenized_dataset,
	compute_metrics=compute_metrics
)

# 学習実行(推論時はコメントアウト)
trainer.train()