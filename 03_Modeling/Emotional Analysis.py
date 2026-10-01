import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


# 1. 初始化 VADER
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()


# 2. 读取数据
df = pd.read_csv("amazon_cleaned.csv")

print("原始数据行数：", len(df))
print("所有列名：")
print(df.columns.tolist())


# 3. 提取评论文本
# 如果你的评论列不叫 review_content，
# 把这里改成你真实的列名
review_column = "review_content"

reviews_df = df[[review_column]].copy()

# 删除评论为空的记录
reviews_df = reviews_df.dropna(subset=[review_column])

# 删除完全空白的字符串
reviews_df = reviews_df[
    reviews_df[review_column].astype(str).str.strip() != ""
].copy()

print("有效评论数量：", len(reviews_df))


# 4. 情感打分函数
def get_sentiment_score(text):
    scores = sia.polarity_scores(str(text))
    return scores["compound"]


# 5. 情感分类函数
def classify_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


# 6. 对所有评论进行情感分析
reviews_df["sentiment_score"] = reviews_df[review_column].apply(
    get_sentiment_score
)

reviews_df["sentiment"] = reviews_df["sentiment_score"].apply(
    classify_sentiment
)


# 7. 查看分析结果
print("\n前 10 条分析结果：")

print(
    reviews_df[
        [review_column, "sentiment_score", "sentiment"]
    ].head(10)
)


# 8. 统计 Positive / Neutral / Negative
sentiment_counts = reviews_df["sentiment"].value_counts()

print("\n情感数量：")
print(sentiment_counts)


# 9. 计算情感比例
sentiment_percent = (
    reviews_df["sentiment"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\n情感比例：")
print(sentiment_percent)


# 10. 保存全部情感分析结果
result_df = df[
    [
        "product_id",
        "product_name",
        "category",
        "rating",
        "review_content"
    ]
].copy()

result_df["sentiment_score"] = reviews_df["sentiment_score"].values
result_df["sentiment"] = reviews_df["sentiment"].values

result_df.to_csv(
    "amazon_sentiment_analysis_full.csv",
    index=False,
    encoding="utf-8-sig"
)

print("已保存：amazon_sentiment_analysis_full.csv")


# 11. 另外生成去重后的评论数据
# 后面做 LDA 可以用
unique_reviews_df = reviews_df.drop_duplicates(
    subset=[review_column]
).copy()

unique_reviews_df.to_csv(
    "amazon_unique_reviews_sentiment.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "去重前评论数：",
    len(reviews_df)
)

print(
    "去重后评论数：",
    len(unique_reviews_df)
)

print(
    "已保存：amazon_unique_reviews_sentiment.csv"
)

for i in range(5):
    print("ROW", i)

    print("review_id:")
    print(df.loc[i, "review_id"])

    print("\nreview_title:")
    print(df.loc[i, "review_title"])

    print("\nreview_content:")
    print(df.loc[i, "review_content"])


# 12. 保留商品级分析需要的字段
product_sentiment = df[
    [
        "product_id",
        "product_name",
        "category",
        "rating",
        "review_content"
    ]
].copy()

# 把刚才算好的 sentiment_score 和 sentiment 加回来
product_sentiment["sentiment_score"] = reviews_df["sentiment_score"].values
product_sentiment["sentiment"] = reviews_df["sentiment"].values


# 13. 找最正面的商品 Top 10
most_positive = product_sentiment.sort_values(
    by="sentiment_score",
    ascending=False
).head(10)

print("Positive Sentiment Top 10")

print(
    most_positive[
        [
            "product_name",
            "rating",
            "sentiment_score",
            "sentiment"
        ]
    ].to_string(index=False)
)


# 14. 找最负面的商品 Top 10
most_negative = product_sentiment.sort_values(
    by="sentiment_score",
    ascending=True
).head(10)

print("Negative Sentiment Top 10")

print(
    most_negative[
        [
            "product_name",
            "rating",
            "sentiment_score",
            "sentiment"
        ]
    ].to_string(index=False)
)


# 15. 保存两个榜单
most_positive.to_csv(
    "top10_positive_products.csv",
    index=False,
    encoding="utf-8-sig"
)

most_negative.to_csv(
    "top10_negative_products.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n已保存：top10_positive_products.csv")
print("已保存：top10_negative_products.csv")