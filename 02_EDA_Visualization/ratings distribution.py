import pandas as pd
import matplotlib.pyplot as plt


# 1. 文件路径与参数

file_path = "ratings_Electronics (1).csv"
chunk_size = 500_000


# 2. 初始化统计变量

# 提前建立1～5分的计数器
rating_counts = pd.Series(
    0,
    index=[1, 2, 3, 4, 5],
    dtype="int64"
)

total_rows = 0
valid_rating_rows = 0
rating_sum = 0.0


# 3. 分块读取，并手动添加表头

reader = pd.read_csv(
    file_path,
    header=None,
    names=[
        "user_id",
        "product_id",
        "rating",
        "timestamp"
    ],
    usecols=["rating"],
    dtype={
        "rating": "float32"
    },
    chunksize=chunk_size
)


# 4. 分块统计评分

for chunk_number, chunk in enumerate(reader, start=1):

    total_rows += len(chunk)

    # 将异常值转换为空值
    chunk["rating"] = pd.to_numeric(
        chunk["rating"],
        errors="coerce"
    )

    # 只保留1～5分
    valid_rating = chunk.loc[
        chunk["rating"].between(1, 5),
        "rating"
    ].dropna()

    valid_rating_rows += len(valid_rating)
    rating_sum += valid_rating.sum()

    # 统计当前数据块中每个评分的数量
    chunk_counts = (
        valid_rating
        .astype("int8")
        .value_counts()
    )

    # 累加到总计数器
    rating_counts = rating_counts.add(
        chunk_counts,
        fill_value=0
    ).astype("int64")

    print(
        f"已处理第 {chunk_number} 个数据块，"
        f"累计读取 {total_rows:,} 行"
    )


# 按照评分从低到高排列
rating_counts = rating_counts.sort_index()


# 5. 计算评分比例

rating_distribution = pd.DataFrame({
    "rating": rating_counts.index,
    "rating_count": rating_counts.values
})

rating_distribution["percentage"] = (
    rating_distribution["rating_count"]
    / valid_rating_rows
)

rating_distribution["percentage_percent"] = (
    rating_distribution["percentage"] * 100
).round(2)


# 6. 计算平均数、中位数和众数

average_rating = rating_sum / valid_rating_rows

cumulative_counts = rating_counts.cumsum()


# 找到指定位置对应的评分
def rating_at_position(position):

    return cumulative_counts[
        cumulative_counts > position
    ].index[0]


# 使用从0开始的位置计算中位数
middle_position_1 = (valid_rating_rows - 1) // 2
middle_position_2 = valid_rating_rows // 2

median_rating = (
    rating_at_position(middle_position_1)
    + rating_at_position(middle_position_2)
) / 2


# 出现次数最多的评分
mode_rating = rating_counts.idxmax()


# 7. 输出统计结果

print("\n评分大表总行数：", f"{total_rows:,}")
print("有效评分数量：", f"{valid_rating_rows:,}")
print("无效或缺失评分数量：", f"{total_rows - valid_rating_rows:,}")

print("\n整体评分分布：")
print(
    rating_distribution[
        [
            "rating",
            "rating_count",
            "percentage_percent"
        ]
    ].to_string(index=False)
)

print("\n平均评分：", round(average_rating, 4))
print("评分中位数：", median_rating)
print("评分众数：", mode_rating)


# 8. 保存评分分布表

rating_distribution.to_csv(
    "rating_distribution.csv",
    index=False
)


# 9. 绘制评分分布柱状图

plt.figure(figsize=(9, 6))

bars = plt.bar(
    rating_distribution["rating"],
    rating_distribution["rating_count"],
    color=[
        "firebrick",
        "darkorange",
        "gold",
        "cornflowerblue",
        "mediumseagreen"
    ],
    edgecolor="black",
    alpha=0.85
)

plt.title("Overall Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Ratings")

plt.xticks([1, 2, 3, 4, 5])
plt.grid(axis="y", alpha=0.25)


# 在每根柱子上标注数量和比例
for bar, count, percentage in zip(
    bars,
    rating_distribution["rating_count"],
    rating_distribution["percentage_percent"]
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{count:,}\n({percentage:.2f}%)",
        ha="center",
        va="bottom",
        fontsize=9
    )


plt.tight_layout()

plt.savefig(
    "overall_rating_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()