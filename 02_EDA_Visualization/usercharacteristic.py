import math
import pandas as pd
import matplotlib.pyplot as plt


# 1. 文件路径与参数

file_path = "ratings_Electronics (1).csv"

# 每次读取50万行，避免一次读取全部数据导致内存不足
chunk_size = 500_000


# 2. 分块读取并手动添加表头

reader = pd.read_csv(
    file_path,
    header=None,
    names=[
        "user_id",
        "product_id",
        "rating",
        "timestamp"
    ],
    dtype={
        "user_id": "string",
        "product_id": "string",
        "rating": "float32",
        "timestamp": "int64"
    },
    chunksize=chunk_size
)


# 用于保存各数据块的用户汇总结果
user_summary = None
total_rows = 0


# 3. 按照user_id汇总每一块数据

for chunk_number, chunk in enumerate(reader, start=1):

    total_rows += len(chunk)

    # 删除关键字段缺失的数据
    chunk = chunk.dropna(
        subset=[
            "user_id",
            "product_id",
            "rating",
            "timestamp"
        ]
    ).copy()

    # 好评：评分大于等于4分
    chunk["positive_rating"] = (
        chunk["rating"] >= 4
    ).astype("int8")

    # 差评：评分小于等于2分
    chunk["negative_rating"] = (
        chunk["rating"] <= 2
    ).astype("int8")

    # 对当前数据块按照用户汇总
    chunk_summary = (
        chunk
        .groupby("user_id")
        .agg(
            rating_count=(
                "rating",
                "size"
            ),
            rating_sum=(
                "rating",
                "sum"
            ),
            positive_count=(
                "positive_rating",
                "sum"
            ),
            negative_count=(
                "negative_rating",
                "sum"
            ),
            first_timestamp=(
                "timestamp",
                "min"
            ),
            last_timestamp=(
                "timestamp",
                "max"
            )
        )
    )

    # 合并不同数据块中的同一位用户
    if user_summary is None:
        user_summary = chunk_summary
    else:
        user_summary = (
            pd.concat(
                [
                    user_summary,
                    chunk_summary
                ]
            )
            .groupby(level=0)
            .agg(
                rating_count=(
                    "rating_count",
                    "sum"
                ),
                rating_sum=(
                    "rating_sum",
                    "sum"
                ),
                positive_count=(
                    "positive_count",
                    "sum"
                ),
                negative_count=(
                    "negative_count",
                    "sum"
                ),
                first_timestamp=(
                    "first_timestamp",
                    "min"
                ),
                last_timestamp=(
                    "last_timestamp",
                    "max"
                )
            )
        )

    print(
        f"已处理第 {chunk_number} 个数据块，"
        f"累计读取 {total_rows:,} 行"
    )


print("\nratings大表总行数：", f"{total_rows:,}")


# 4. 计算每位用户的行为指标

user_summary = user_summary.reset_index()


# 该ratings数据中，每行代表一个用户对一个商品的评分
# 因此评价次数可以作为评价商品数量使用
user_summary["rated_product_count"] = (
    user_summary["rating_count"]
)


# 用户平均评分
user_summary["average_rating"] = (
    user_summary["rating_sum"]
    / user_summary["rating_count"]
)


# 好评率：4分和5分所占比例
user_summary["positive_rate"] = (
    user_summary["positive_count"]
    / user_summary["rating_count"]
)


# 差评率：1分和2分所占比例
user_summary["negative_rate"] = (
    user_summary["negative_count"]
    / user_summary["rating_count"]
)


# 转换首次和最近评价时间
user_summary["first_rating_date"] = pd.to_datetime(
    user_summary["first_timestamp"],
    unit="s"
)

user_summary["last_rating_date"] = pd.to_datetime(
    user_summary["last_timestamp"],
    unit="s"
)


# 用户活跃时间跨度
user_summary["active_days"] = (
    user_summary["last_rating_date"]
    - user_summary["first_rating_date"]
).dt.days


# 保留最终分析需要的字段
user_summary = user_summary[
    [
        "user_id",
        "rating_count",
        "rated_product_count",
        "average_rating",
        "positive_rate",
        "negative_rate",
        "first_rating_date",
        "last_rating_date",
        "active_days"
    ]
]


# 对小数进行四舍五入
user_summary[
    [
        "average_rating",
        "positive_rate",
        "negative_rate"
    ]
] = user_summary[
    [
        "average_rating",
        "positive_rate",
        "negative_rate"
    ]
].round(4)


print("唯一用户数：", f"{len(user_summary):,}")

print("\n用户特征汇总表前10行：")
print(user_summary.head(10).to_string(index=False))


# 5. 将排名前10%的用户定义为高活跃用户

user_summary = user_summary.sort_values(
    by=[
        "rated_product_count",
        "active_days"
    ],
    ascending=[
        False,
        False
    ]
).reset_index(drop=True)


# 计算前10%的用户数量
high_active_user_number = math.ceil(
    len(user_summary) * 0.10
)


# 默认所有用户属于普通用户
user_summary["user_group"] = "Regular User"


# 排名前10%的用户定义为高活跃用户
user_summary.loc[
    :high_active_user_number - 1,
    "user_group"
] = "High-activity User"


print(
    "\n高活跃用户数量：",
    f"{high_active_user_number:,}"
)

print(
    "普通用户数量：",
    f"{len(user_summary) - high_active_user_number:,}"
)


# 查看高活跃用户的最低评价商品数量
high_activity_minimum = user_summary.loc[
    high_active_user_number - 1,
    "rated_product_count"
]

print(
    "进入前10%至少需要评价的商品数量：",
    high_activity_minimum
)


# 6. 比较高活跃用户与普通用户

group_comparison = (
    user_summary
    .groupby(
        "user_group",
        sort=False
    )
    .agg(
        user_count=(
            "user_id",
            "nunique"
        ),
        average_rated_products=(
            "rated_product_count",
            "mean"
        ),
        median_rated_products=(
            "rated_product_count",
            "median"
        ),
        average_rating=(
            "average_rating",
            "mean"
        ),
        average_positive_rate=(
            "positive_rate",
            "mean"
        ),
        average_negative_rate=(
            "negative_rate",
            "mean"
        ),
        average_active_days=(
            "active_days",
            "mean"
        ),
        median_active_days=(
            "active_days",
            "median"
        )
    )
    .reset_index()
    .round(4)
)


print("\n高活跃用户与普通用户对比表：")
print(group_comparison.to_string(index=False))


# 7. 输出评价商品数量最多的20位用户

top_20_users = (
    user_summary
    .head(20)
    [
        [
            "user_id",
            "rated_product_count",
            "average_rating",
            "positive_rate",
            "negative_rate",
            "active_days"
        ]
    ]
)


print("\n评价商品数量最多的20位用户：")
print(top_20_users.to_string(index=False))


# 8. 保存统计结果

user_summary.to_csv(
    "user_activity_summary.csv",
    index=False
)

group_comparison.to_csv(
    "user_group_comparison.csv",
    index=False
)

top_20_users.to_csv(
    "top_20_active_users.csv",
    index=False
)


# 9. 绘制两类用户的平均评分

plt.figure(figsize=(8, 6))

plt.bar(
    group_comparison["user_group"],
    group_comparison["average_rating"],
    color=[
        "cornflowerblue",
        "orange"
    ]
)

plt.title("Average Rating by User Group")
plt.xlabel("User Group")
plt.ylabel("Average Rating")

plt.ylim(0, 5)
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

plt.savefig(
    "user_group_average_rating.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# 10. 绘制两类用户的好评率与差评率

plot_data = group_comparison.set_index(
    "user_group"
)[
    [
        "average_positive_rate",
        "average_negative_rate"
    ]
]


plot_data.plot(
    kind="bar",
    figsize=(9, 6),
    color=[
        "mediumseagreen",
        "tomato"
    ]
)

plt.title("Positive and Negative Rating Rates by User Group")
plt.xlabel("User Group")
plt.ylabel("Average Rate")

plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(
    [
        "Positive Rate",
        "Negative Rate"
    ]
)

plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

plt.savefig(
    "user_group_rating_rates.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()