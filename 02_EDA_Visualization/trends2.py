import pandas as pd
import matplotlib.pyplot as plt


# 1. 读取数据

df = pd.read_csv("amazon_cleaned.csv")


# 2. 商品去重

df_product = (
    df.drop_duplicates(
        subset=["product_id"],
        keep="first"
    )
    .copy()
)

print("去重前行数：", len(df))
print("去重后行数：", len(df_product))
print(
    "唯一商品数：",
    df_product["product_id"].nunique()
)


# 3. 提取一级类目

df_product["main_category"] = (
    df_product["category"]
    .str.split("|")
    .str[0]
)


# 4. 计算类目价格汇总表

category_price = (
    df_product
    .groupby("main_category")
    .agg(
        product_count=(
            "product_id",
            "nunique"
        ),
        average_price=(
            "discounted_price",
            "mean"
        ),
        median_price=(
            "discounted_price",
            "median"
        ),
        price_std=(
            "discounted_price",
            "std"
        ),
        minimum_price=(
            "discounted_price",
            "min"
        ),
        q1_price=(
            "discounted_price",
            lambda x: x.quantile(0.25)
        ),
        q3_price=(
            "discounted_price",
            lambda x: x.quantile(0.75)
        ),
        maximum_price=(
            "discounted_price",
            "max"
        )
    )
    .reset_index()
)

category_price = (
    category_price
    .round(2)
    .sort_values(
        "median_price",
        ascending=True
    )
)

print(category_price)

category_price.to_csv(
    "category_price_summary.csv",
    index=False
)

# 筛选样本量足够的类目

valid_category_price = (
    category_price[
        category_price["product_count"] >= 10
    ]
    .copy()
    .sort_values(
        "median_price",
        ascending=True
    )
)

# 在图表类目名称后面显示样本量
valid_category_price["category_label"] = (
    valid_category_price["main_category"]
    + " (n="
    + valid_category_price["product_count"]
      .astype(str)
    + ")"
)

print("用于正式图表的类目：")
print(
    valid_category_price[[
        "main_category",
        "product_count",
        "average_price",
        "median_price"
    ]]
)

# 平均价与中位价对比图

plot_data = (
    valid_category_price
    .set_index("category_label")
    [[
        "average_price",
        "median_price"
    ]]
)

ax = plot_data.plot(
    kind="barh",
    figsize=(11, 7),
    color=[
        "steelblue",
        "orange"
    ]
)

plt.title(
    "Average and Median Discounted Price by Category"
)

plt.xlabel("Discounted Price (INR)")
plt.ylabel("Product Category")

plt.legend([
    "Average Price",
    "Median Price"
])

plt.grid(
    axis="x",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    "category_average_median_price_filtered.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# 类目价格分布箱线图

# 按中位价格从低到高排列类目
category_order = (
    valid_category_price["main_category"]
    .tolist()
)

# 生成带样本量的类目标签
category_labels = (
    valid_category_price["category_label"]
    .tolist()
)

# 分别取出每个类目的全部折扣价
box_data = [
    df_product.loc[
        df_product["main_category"]
        == category,
        "discounted_price"
    ].dropna()
    for category in category_order
]

plt.figure(figsize=(12, 7))

plt.boxplot(
    box_data,
    tick_labels=category_labels,
    orientation="horizontal",
    showfliers=True
)

plt.title(
    "Discounted Price Distribution by Product Category"
)

plt.xlabel("Discounted Price (INR)")
plt.ylabel("Product Category")

plt.grid(
    axis="x",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    "category_price_distribution_boxplot_filtered.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()