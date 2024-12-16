from transformers import AutoTokenizer
import transformers
import torch
import pandas as pd

########
# 真偽判定を行う（テスト段階）
########

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# 真偽判定指示（contentのみ）↓
# queri = "The text enclosed in [ ] below is the body of the news article. Please determine whether this news is true or fake. If it is true, please output ""Real"" at the beginning of your answer, and if it is fake, please output ""Fake"".\n"

# 真偽判定指示（title + content）↓
queri = "The part enclosed in { } below is the title of the news article, and the part enclosed in [ ] is the main text of the news article. Based on these two factors, if the news content is true, please output ""Real"" at the beginning of your answer, and if it is fake news, please output ""Fake"".\n"
# 古いやつ↓
# queri = "The part enclosed in { } below is the title of the news article, and the part enclosed in [ ] is the main text of the news article. Based on these two factors, if it is true, please output ""Real"" at the beginning of your answer, and if it is fake, please output ""Fake"".\n"

def prompt_template_31(title, text):
    prompt = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are an expert in spotting fake news.<|eot_id|><|start_header_id|>user<|end_header_id|>

    #The news title and text are shown below. Based on these two factors, if the news content is true, please output ""Real"" at the beginning of your answer, and if it is fake news, please output ""Fake"".
    #News Data
        ##title: {title}
        ##text: {text} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    return prompt

def main(df, row_num):
    prediction = []  # 予測値
    true_label = []  # 真値
    count = 0  # 間違えた回数
    count_t = 0  # trueを当てた回数
    count_f = 0  # fakeを当てた回数
    violation_text = []
    count_violation = 0  # 違反回数

    print("行数：", row_num)

    for i in range(row_num):
        title = df.at[i, "title"]
        text = df.at[i, "content"]
        # text = df.at[i, "text"]
        label = df.at[i, "label"]

        true_label.append(label)

        # Llama3への入力文（contentのみ）
        # inputs = queri + "[" + str(text) + "]"

        # Llama3への入力文（title + content）
        # inputs = queri + "{" + str(title) + "}" + "[" + str(text) + "]"
        inputs = prompt_template_31(str(title), str(text))

        # print("---------------------------------------------")
        # print("input : ", inputs)
        # print("label : ", label)

        messages = [
            {"role": "system", "content": inputs},
            # {"role": "user", "content": "Who are you?"},
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        get_answer = outputs[0]["generated_text"][-1]["content"]

        print("解答：", get_answer)
        if "Real" in get_answer[:6]:
            prediction.append(0)
            if label == 1:
                # print("間違えた")
                count += 1
            elif label == 0:
                count_t += 1
        elif "Fake" in get_answer[:6]:
            prediction.append(1)
            if label == 0:
                # print("間違えた")
                count += 1
            elif label == 1:
                count_f += 1
        else:
            prediction.append(-1)
            count_violation += 1
            violation_text.append(get_answer[:6])
            print(violation_text)
            print("違反回答：", get_answer[:6])

        # print("---------------------------------------------")

    print("結果：", prediction)
    print("真値ラベル：", true_label)
    print("間違えた回数：", count)
    print("trueを当てた回数：", count_t)
    print("fakeを当てた回数 ：", count_f)
    print("違反回答例：", violation_text)
    print("違反回数：", count_violation)
