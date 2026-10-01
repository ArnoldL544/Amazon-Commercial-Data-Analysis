# 读取加预处理
import pandas as pd
import matplotlib.pyplot as plt


# 1. 读取清洗后的数据

df = pd.read_csv("amazon_cleaned.csv")

print("原始数据大小：", df.shape)
print(df.head())


# 2. 商品去重
# 同一个商品可能对应多条用户评价
# 每个product_id只保留一行

df_product = (
    df.drop_duplicates(
        subset=["product_id"],
        keep="first"
    )
    .copy()
)

print("商品去重后的数据大小：", df_product.shape)


# 3. 提取一级商品类目

df_product["main_category"] = (
    df_product["category"]
    .astype(str)
    .str.split("|")
    .str[0]
    .str.strip()
)


# 4. 转换数字字段

numeric_columns = [
    "actual_price",
    "discounted_price",
    "rating",
    "rating_count"
]

for column in numeric_columns:
    df_product[column] = pd.to_numeric(
        df_product[column],
        errors="coerce"
    )


# 把“64%”转换成数字64
df_product["discount_percentage"] = pd.to_numeric(
    df_product["discount_percentage"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.strip(),
    errors="coerce"
)


# 5. 删除实验所需字段缺失的记录

df_product = df_product.dropna(
    subset=[
        "product_id",
        "main_category"
    ]
)


# 6. 只保留商品数量不少于10的类目

category_count = (
    df_product
    .groupby("main_category")["product_id"]
    .nunique()
)

valid_categories = category_count[
    category_count >= 10
].index

df_valid = df_product[
    df_product["main_category"].isin(valid_categories)
].copy()


print("\n参与实验的商品类目及商品数量：")
print(
    category_count
    .loc[valid_categories]
    .sort_values(ascending=False)
)

print("\n正式分析数据大小：", df_valid.shape)







# 实验1，商品类目 价格 折扣
# 实验一：商品类目 × 价格 × 折扣

category_price_discount = (
    df_valid
    .groupby("main_category")
    .agg(
        product_count=("product_id", "nunique"),
        median_actual_price=("actual_price", "median"),
        median_discounted_price=("discounted_price", "median"),
        average_discount_rate=("discount_percentage", "mean"),
        median_discount_rate=("discount_percentage", "median")
    )
    .reset_index()
    .round(2)
)

print("不同商品类目的价格与折扣：")
print(category_price_discount)


# 按平均折扣率排序，方便画图
discount_plot = category_price_discount.sort_values(
    "average_discount_rate",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    discount_plot["main_category"],
    discount_plot["average_discount_rate"],
    color="mediumseagreen"
)

plt.title("Average Discount Rate by Product Category")
plt.xlabel("Average Discount Rate (%)")
plt.ylabel("Product Category")

plt.grid(axis="x", alpha=0.25)
plt.tight_layout()

plt.savefig(
    "category_average_discount_rate.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()







# 实验2，商品类目 评分 评价热度
# 实验二：商品类目 × 评分 × 评价热度

category_reputation = (
    df_valid
    .groupby("main_category")
    .agg(
        product_count=("product_id", "nunique"),
        average_rating=("rating", "mean"),
        median_rating=("rating", "median"),
        average_rating_count=("rating_count", "mean"),
        median_rating_count=("rating_count", "median")
    )
    .reset_index()
    .round(2)
)

print("不同商品类目的评分与评价热度：")
print(category_reputation)


# 绘制不同类目的平均评分
rating_plot = category_reputation.sort_values(
    "average_rating",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    rating_plot["main_category"],
    rating_plot["average_rating"],
    color="cornflowerblue"
)

plt.title("Average Rating by Product Category")
plt.xlabel("Average Rating")
plt.ylabel("Product Category")

plt.xlim(0, 5)
plt.grid(axis="x", alpha=0.25)
plt.tight_layout()

plt.savefig(
    "category_average_rating.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# 绘制不同类目的评价数量中位数
rating_count_plot = category_reputation.sort_values(
    "median_rating_count",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    rating_count_plot["main_category"],
    rating_count_plot["median_rating_count"],
    color="orange"
)

plt.title("Median Rating Count by Product Category")
plt.xlabel("Median Rating Count")
plt.ylabel("Product Category")

plt.grid(axis="x", alpha=0.25)
plt.tight_layout()

plt.savefig(
    "category_median_rating_count.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()