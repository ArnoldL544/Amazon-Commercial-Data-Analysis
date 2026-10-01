# 1. 导入需要的库

import re
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from wordcloud import WordCloud, STOPWORDS


# 2. 读取数据

file_path = "amazon_cleaned.csv"

df = pd.read_csv(file_path)

print("原始数据行数：", len(df))


# 3. 按商品编号去重
# 避免同一商品的评论文本被重复统计

df_product = (
    df.drop_duplicates(
        subset=["product_id"],
        keep="first"
    )
    .copy()
)

print("商品去重后行数：", len(df_product))
print("唯一商品数：", df_product["product_id"].nunique())


# 4. 提取并合并评论文本

review_series = (
    df_product["review_content"]
    .dropna()
    .astype(str)
    .str.strip()
)

# 删除空文本
review_series = review_series[
    review_series != ""
]

print("包含评论文本的商品数：", len(review_series))


# 将所有商品的评论文本合并
review_text = " ".join(review_series)


# 5. 设置停用词

stopwords = set(STOPWORDS)

# 添加没有太大分析价值的常见词
custom_stopwords = {
    "amazon",
    "product",
    "products",
    "item",
    "items",
    "buy",
    "buying",
    "bought",
    "purchase",
    "purchased",
    "one",
    "use",
    "using",
    "used",
    "get",
    "gets",
    "getting",
    "got",
    "really",
    "also",
    "much",
    "thing",
    "things",
    "not",
    "no",
    "never",
    "will"
}

stopwords.update(custom_stopwords)


# 6. 清理文本并提取英文单词

# 转成小写并提取英文单词
words = re.findall(
    r"\b[a-zA-Z']+\b",
    review_text.lower()
)


# 清理单词两侧可能出现的撇号
words = [
    word.strip("'")
    for word in words
]


# 删除停用词、空词和长度小于等于2的词
filtered_words = [
    word
    for word in words
    if word
    and word not in stopwords
    and len(word) > 2
]


print("清理前单词总数：", f"{len(words):,}")
print("清理后单词总数：", f"{len(filtered_words):,}")


# 7. 使用Counter统计真实词频

word_counts = Counter(filtered_words)


# 检查是否存在可用于生成词云的词
if len(word_counts) == 0:

    raise ValueError(
        "文本清理后没有剩余词语，无法生成词云。"
    )


print("不同词语数量：", f"{len(word_counts):,}")


# 8. 生成完整词频表

word_frequency = pd.DataFrame(
    word_counts.most_common(),
    columns=[
        "word",
        "count"
    ]
)


# 保存所有词语的真实出现次数
word_frequency.to_csv(
    "review_word_frequency.csv",
    index=False
)


# 9. 生成前20个高频词表

top_20_words = word_frequency.head(20).copy()


print("\n评论中出现次数最多的20个词：")

print(
    top_20_words.to_string(
        index=False
    )
)


top_20_words.to_csv(
    "top_20_review_words.csv",
    index=False
)


# 10. 根据Counter词频生成词云

wordcloud = WordCloud(
    width=1200,
    height=700,
    background_color="white",
    max_words=150,
    colormap="viridis",
    collocations=False,
    random_state=42
).generate_from_frequencies(word_counts)


# 11. 绘制词云

plt.figure(figsize=(12, 7))

plt.imshow(
    wordcloud,
    interpolation="bilinear"
)

plt.axis("off")

plt.title(
    "Amazon Review Word Cloud",
    fontsize=16
)

plt.tight_layout()


# 12. 保存词云图片

plt.savefig(
    "amazon_review_wordcloud.png",
    dpi=300,
    bbox_inches="tight"
)


# 13. 最后显示词云

plt.show()