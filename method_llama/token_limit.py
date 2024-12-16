import pandas as pd
from transformers import AutoTokenizer

#########
# トークンを指定した数までカット、もしくは超過したデータを削除する関数
########

# トークナイザーの初期化
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # ここはLlama2やLlama3のモデル名を指定
tokenizer = AutoTokenizer.from_pretrained(model_name)

# トークン数の上限を設定
TOKEN_LIMIT = 1024  # 例として1024トークン

def main(df, text_column, token_limit=TOKEN_LIMIT, action='cut'):
    """
    DataFrame内のテキストデータをトークン数で処理。

    Parameters:
    - df: pandas.DataFrame - 入力データフレーム
    - text_column: str - テキストが含まれるカラム名
    - token_limit: int - トークン数の上限
    - action: str - 'cut'（トークン数を超えた場合にカット）または 'remove'（行を削除）

    Returns:
    - pandas.DataFrame - 処理後のデータフレーム
    """
    def tokenize_text(text):
        return tokenizer.encode(text, add_special_tokens=False)
    
    # トークン数を計算
    df['token_count'] = df[text_column].apply(lambda x: len(tokenize_text(str(x))))
    
    if action == 'cut':
        def cut_text(text):
            tokens = tokenize_text(str(text))
            if len(tokens) > token_limit:
                tokens = tokens[:token_limit]
                return tokenizer.decode(tokens)
            return text
        df[text_column] = df[text_column].apply(cut_text)   # テキストをカット
        df['token_count'] = df[text_column].apply(lambda x: len(tokenize_text(str(x)))) # カット後のトークン数を再計算
        df = df[df['token_count'] <= token_limit]  # 必要に応じてオプション
    elif action == 'remove':
        df = df[df['token_count'] <= token_limit]
    else:
        raise ValueError("actionパラメータは 'cut' か 'remove' を指定してください。")
    
    # トークン数のカラムを削除
    df = df.drop(columns=['token_count'])
    
    return df


"""
# 使用例

# サンプルデータの作成
data = {
    'id': [1, 2, 3],
    'text': [
        "これは短いテキストです。",
        "これは非常に長いテキストであり、トークン数が多くなります。" * 100,  # トークン数が多い
        "適切な長さのテキストです。"
    ]
}

df = pd.DataFrame(data)

print("元のデータフレーム:")
print(df)

# テキストをカットする場合
df_cut = main(df.copy(), text_column='text', token_limit=50, action='cut')
print("\nトークン数をカットしたデータフレーム:")
print(df_cut)

# トークン数が超過した行を削除する場合
df_remove = main(df.copy(), text_column='text', token_limit=50, action='remove')
print("\nトークン数が超過した行を削除したデータフレーム:")
print(df_remove)



# 以下はKaggleデータセットでテスト

df = pd.read_csv("../sample_Kaggle_train_100.csv")

df = df.drop(["id", "author", "title", "content_cleaned"],  axis=1)

print("元のデータフレーム:")
print(df)

# テキストをカットする場合
df_cut = main(df.copy(), text_column='text', token_limit=5, action='cut')
print("\nトークン数をカットしたデータフレーム:")
print(df_cut)

# トークン数が超過した行を削除する場合
df_remove = main(df.copy(), text_column='text', token_limit=500, action='remove')
print("\nトークン数が超過した行を削除したデータフレーム:")
print(df_remove)
"""