import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 1. 读取数据

file_path = "amazon_cleaned.csv"

df = pd.read_csv(file_path)

print("读取数据行数：", len(df))


# 2. 按商品编号去重

df_product = (
    df.drop_duplicates(
        subset=["product_id"],
        keep="first"
    )
    .copy()
)

print("商品去重后行数：", len(df_product))
print("唯一商品数：", df_product["product_id"].nunique())


# 3. 清理评分和评论文本

df_product["rating"] = pd.to_numeric(
    df_product["rating"],
    errors="coerce"
)


# 删除评分或评论正文缺失的数据
df_review = df_product.dropna(
    subset=[
        "product_id",
        "rating",
        "review_content"
    ]
).copy()


# 只保留1～5分
df_review = df_review[
    df_review["rating"].between(1, 5)
].copy()


# 清理评论文本
df_review["review_content"] = (
    df_review["review_content"]
    .astype(str)
    .str.strip()
)


# 删除空评论
df_review = df_review[
    df_review["review_content"] != ""
].copy()

print("参与分析的商品数：", len(df_review))


# 4. 计算评论文本长度

# 字符数量：包括字母、数字、空格和标点
df_review["review_character_count"] = (
    df_review["review_content"]
    .str.len()
)


# 英文单词数量
df_review["review_word_count"] = (
    df_review["review_content"]
    .str.findall(r"\b[\w']+\b")
    .str.len()
)


print("\n评论文本长度基本统计：")

print(
    df_review[
        [
            "review_character_count",
            "review_word_count"
        ]
    ]
    .describe()
    .round(2)
)


# 5. 将商品综合评分分成五组

group_order = [
    "0-1",
    "1-2",
    "2-3",
    "3-4",
    "4-5"
]


df_review["rating_group"] = pd.cut(
    df_review["rating"],
    bins=[
        0,
        1,
        2,
        3,
        4,
        5
    ],
    labels=group_order,
    right=True,
    include_lowest=True
)


print("\n各评分区间的商品数量：")

print(
    df_review["rating_group"]
    .value_counts(sort=False)
)


# 6. 按评分区间统计评论文本长度

review_length_summary = (
    df_review
    .groupby(
        "rating_group",
        observed=False
    )
    .agg(
        product_count=(
            "product_id",
            "nunique"
        ),

        average_rating=(
            "rating",
            "mean"
        ),

        average_character_count=(
            "review_character_count",
            "mean"
        ),

        median_character_count=(
            "review_character_count",
            "median"
        ),

        average_word_count=(
            "review_word_count",
            "mean"
        ),

        median_word_count=(
            "review_word_count",
            "median"
        )
    )
    .reset_index()
    .round(2)
)


print("\n不同评分区间的评论文本长度：")

print(
    review_length_summary.to_string(
        index=False
    )
)


# 7. 计算商品评分与评论文本长度的相关系数

character_correlation = (
    df_review["rating"]
    .corr(
        df_review["review_character_count"]
    )
)

word_correlation = (
    df_review["rating"]
    .corr(
        df_review["review_word_count"]
    )
)


print(
    "\n商品综合评分与评论字符数的相关系数：",
    round(character_correlation, 4)
)

print(
    "商品综合评分与评论单词数的相关系数：",
    round(word_correlation, 4)
)


# 8. 保存分析结果

df_review[
    [
        "product_id",
        "rating",
        "rating_group",
        "review_character_count",
        "review_word_count"
    ]
].to_csv(
    "review_length_data.csv",
    index=False
)


review_length_summary.to_csv(
    "review_length_rating_summary.csv",
    index=False
)


# 9. 绘制五个评分组的点状图和平均趋势线

group_positions = {
    group: position
    for position, group in enumerate(
        group_order,
        start=1
    )
}


# 将评分组转换成横坐标1～5
x_positions = (
    df_review["rating_group"]
    .astype(str)
    .map(group_positions)
    .astype(float)
)


# 加入少量随机偏移，避免点完全重叠
rng = np.random.default_rng(42)

x_jitter = x_positions + rng.normal(
    loc=0,
    scale=0.06,
    size=len(df_review)
)


# 计算每个评分区间的平均评论单词数
group_average_word_count = (
    df_review
    .groupby(
        "rating_group",
        observed=False
    )["review_word_count"]
    .mean()
    .reindex(group_order)
)


plt.figure(figsize=(10, 6))


# 绘制所有商品的点
plt.scatter(
    x_jitter,
    df_review["review_word_count"],
    color="cornflowerblue",
    alpha=0.30,
    s=25,
    label="Products"
)


# 绘制各评分区间平均值趋势线
plt.plot(
    range(1, 6),
    group_average_word_count.values,
    color="red",
    marker="o",
    markersize=7,
    linewidth=2.5,
    label="Average Review Length"
)


plt.title(
    "Review Text Length by Product Rating Group"
)

plt.xlabel("Product Rating Group")
plt.ylabel("Review Text Word Count")

plt.xticks(
    range(1, 6),
    group_order
)

plt.xlim(0.5, 5.5)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "rating_group_review_length_scatter.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# 10. 绖制五个评分组的箱线图

boxplot_data = []


for group in group_order:

    group_values = df_review.loc[
        df_review["rating_group"].astype(str) == group,
        "review_word_count"
    ].to_numpy()

    # 如果某一评分区间没有商品，保留空白位置
    if len(group_values) == 0:
        group_values = np.array([np.nan])

    boxplot_data.append(group_values)


plt.figure(figsize=(10, 6))

plt.boxplot(
    boxplot_data,
    tick_labels=group_order,
    showfliers=False,
    patch_artist=True,

    boxprops={
        "facecolor": "lightblue",
        "edgecolor": "black"
    },

    medianprops={
        "color": "red",
        "linewidth": 2
    }
)

plt.title(
    "Review Text Length Distribution by Product Rating Group"
)

plt.xlabel("Product Rating Group")
plt.ylabel("Review Text Word Count")

plt.grid(axis="y", alpha=0.25)

plt.tight_layout()

plt.savefig(
    "review_length_by_rating_group.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()