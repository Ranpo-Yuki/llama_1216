from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import random
from datasets import Dataset
from datetime import datetime

import method_llama.token_limit    # 自作（一定のトークンを超える記事を削除or切り取り）
# import create_dataset_ISOT  # 自作（ISOTデータセットを前処理する）

########
# モデルとトークナイザーのロード
########
# QLoRA?
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Acceleratorの初期化
accelerator = Accelerator()

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # ここはLlama2やLlama3のモデル名を指定
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config = bnb_config, device_map="auto")

# encoded_input = tokenizer.encode()

# GPUの割当確認↓
for name, param in model.named_parameters():
    print(f"{name}: {param.device}")


########
# LoRA設定
########
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)



########
# PEFTモデルの取得
########
peft_model = get_peft_model(model, lora_config)



########
# データセットのロード
########
# df = pd.read_csv("sample_Kaggle_train_100.csv")   #　お試し用100サンプル
df = pd.read_csv("../Datasets/KaggleFakeNewsDetectionDataset/train.csv")   # Kaggleデータセット
# df = create_dataset_ISOT.main("../ISOTFakeNewsDataset/True.csv", "../ISOTFakeNewsDataset/Fake.csv", r_seed=1)   # ISOTデータセット

"""
#######
# ISOTデータセット用の処理部分
# data_typeごとにデータフレームを分割
grouped = df.groupby('data_type')
edited_groups = []
for data_type, group in grouped:
    edited_groups.append(group)
    print("append：" + data_type)

df_test = edited_groups[0]  
df_train = edited_groups[1]
print(df_train)
print(df_train["label"].value_counts())   #ラベル数確認用
print(df_train["data_type"].value_counts())
print(df_test)
print(df_test["data_type"].value_counts())
# print(df_test["label"].value_counts())    #ラベル数確認用

df = df_train.sample(frac=1, random_state=1) 
########
"""

# 前処理
df = method_llama.token_limit.main(df.copy(), text_column='text', token_limit=2048, action='remove')
df["label"] = df["label"].replace({0: "Real", 1:"fake"})
print("----------------------------------")
print(df)
print("----------------------------------")

# df=pd.read_csv("231228best_reason_record.csv")

dataset=df.to_dict(orient="records")
#######
# orient="records"の挙動例↓
"""
[
    {"id": 1, "name": "Alice", "age": 30},
    {"id": 2, "name": "Bob", "age": 25}
]
"""
#######



##########
# 渡されたデータを元に、特定のフォーマットで文字列を生成
##########
def gen_compound_text(article_record, prefix="Example"):
    title = article_record["title"]
    text = article_record["text"]
    label = article_record["label"]
    prompt=f"""
    #From the following news title and text, output “label” as “Real” if the news content is true, or “Fake” if the news content is fake.
    #News Data
    ##title: {title}
    ##text: {text} 
    ##label : {label}
    """
    return prompt

def prompt_template(article_record):
    title = article_record["title"]
    text = article_record["text"]
    label = article_record["label"]
    prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert in spotting fake news.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
#From the following news title and text, output “label” as “Real” if the news content is true, or “Fake” if the news content is fake.
#News Data
    ##title: {title}
    ##text: {text} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
{label}<|eot_id|>
    """

    return prompt

##########
# 与えられたデータセットから指定された数のトレーニングプロンプトと1つのテストプロンプトを生成するために使用
# 現状未使用
##########
def generate_question_prompt(dataset,test_id,n_prompt_examples=5):
    train_ids=[i for i in range(len(dataset))]
    train_ids.remove(test_id)
    prompt=""

    #train prompt
    for _ in range(n_prompt_examples):
        id=random.choice(train_ids)
        prompt+=gen_compound_text(dataset[id],
                                reason=dataset[id]["Reason"],
                                prediction=dataset[id]["Prediction(integer)"])
        prompt+="\n"

    #test prompt
    prompt+=gen_compound_text(dataset[test_id],prefix="Test")
    #    prompt+="""
    ##Output: Reason, Prediction
    #    """

    return prompt

# 与えられたテキストリストをシャッフルし、トークナイザーを使用してトークン化したデータセットを生成
def prepare_dataset(context_list, tokenizer):
    data_list = [{"text": i} for i in context_list]
    random.shuffle(data_list)

    # tokenize
    dataset = Dataset.from_dict(
        {"text": [item["text"] for item in data_list[:]]})
    
    dataset = dataset.map(lambda samples: tokenizer(samples['text']), batched=True)

    return dataset


#とりあえず初めの10件をテストデータにする
# n_test=10

train_text_list=[]
for id in range(len(dataset)):
    prompt=prompt_template(dataset[id])
    train_text_list.append(prompt)
tokenized_dataset = prepare_dataset(train_text_list, tokenizer)
# tokenized_dataset = prepare_dataset(train_text_list[n_test:], tokenizer)
# tokenized_dataset_test = prepare_dataset(train_text_list[:n_test], tokenizer)

dataset_size = len(tokenized_dataset)   # TrainingArgumentsのlogging_stepsに使用（epochごとにログを出してほしい）

print(tokenized_dataset)                # データ構造確認
print(tokenized_dataset[0]["text"])       # 生データ一個表示


########
# トレーニングの設定
########
tokenizer.pad_token = tokenizer.eos_token
train_args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        num_train_epochs=10,
        learning_rate=1e-5,
        fp16=True,
        logging_steps=dataset_size,
        save_total_limit=1,
        output_dir='log_llama3_outputs/'+datetime.now().strftime('%Y%m%d%H%M%S'),
    )

trainer = transformers.Trainer(
    model=peft_model,
    train_dataset=tokenized_dataset,
    args=train_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

training_result = trainer.train()
print("Training Loss:", training_result.training_loss)

peft_model.save_pretrained("./output_model_llama3_peft/"+datetime.now().strftime('%Y%m%d%H%M%S'))  # 保存するパスを指定
tokenizer.save_pretrained("./output_tokenizer_llama3_peft/"+datetime.now().strftime('%Y%m%d%H%M%S'))    # トークナイザーの保存
