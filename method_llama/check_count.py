from transformers import AutoTokenizer
import transformers
import torch
import pandas as pd

# 同じ判定を行った場合、どれだけ答えがブレるか
# 引数　llama3のパイプラインmessages(messages), データのラベル(label), 何回for文を回すか(max_range)
# Meta-Llama-3-8B-Instruct
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"


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

def main(messages, label, max_range):
    prediction = [] # 予測値
    count = 0 # 間違えた回数
    violation_text = []
    count_violation = 0 # 違反回数

    for i in range(max_range):
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
        if "Real" in get_answer:
            prediction.append(0)
            if label==1:
                print("間違えた")
                count+=1
        elif "Fake" in get_answer:
            prediction.append(1)
            if label==0:
                print("間違えた")
                count+=1
        else:
            prediction.append(-1)
            count_violation += 1
            violation_text.append(get_answer)
            print("違反回答：", get_answer)
    print("結果：", prediction)
    print("間違えた回数：", count)
    print("違反回答例：", violation_text)
    print("違反回数：", count_violation)