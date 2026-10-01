import pandas as pd
import matplotlib.pyplot as plt


# 1. 读取情感分析结果
df = pd.read_csv("amazon_sentiment_analysis_full.csv")

print("数据行数：", len(df))
print("列名：")
print(df.columns.tolist())

# 2. 情感分布百分比
sentiment_percentage = (
    df["sentiment"]
    .value_counts(normalize=True)
    .mul(100)
)

plt.figure(figsize=(7, 5))

plt.bar(
    sentiment_percentage.index,
    sentiment_percentage.values
)

plt.xlabel("Sentiment")
plt.ylabel("Percentage (%)")
plt.title("Overall Sentiment Distribution")

plt.tight_layout()

plt.savefig(
    "sentiment_distribution_percentage.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# 3. Rating vs Sentiment Score
plt.figure(figsize=(8, 6))

plt.scatter(
    df["rating"],
    df["sentiment_score"],
    alpha=0.5
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Product Rating")
plt.ylabel("VADER Sentiment Score")
plt.title("Product Rating vs Review Sentiment")

plt.tight_layout()

plt.savefig(
    "rating_vs_sentiment.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# 4. 商品去重
df_unique = df.drop_duplicates(
    subset=["product_id"]
).copy()


# 5. 找负面情感最强的 Top 10
top_negative = (
    df_unique
    .sort_values(
        by="sentiment_score",
        ascending=True
    )
    .head(10)
    .copy()
)

# 商品名太长，所以截短一点
top_negative["short_name"] = (
    top_negative["product_name"]
    .astype(str)
    .str.slice(0, 45)
)


# 6. Negative Top 10 柱状图
plt.figure(figsize=(10, 7))

plt.barh(
    top_negative["short_name"],
    top_negative["sentiment_score"]
)

plt.xlabel("VADER Sentiment Score")
plt.ylabel("Product")
plt.title("Top 10 Products with Most Negative Sentiment")

plt.tight_layout()

plt.savefig(
    "top10_negative_products.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# 7. 输出 Top 10 方便写报告
print("\nTop 10 Most Negative Products:")

print(
    top_negative[
        [
            "product_name",
            "rating",
            "sentiment_score",
            "sentiment"
        ]
    ].to_string(index=False)
)

print("\n三张图已保存完成。")