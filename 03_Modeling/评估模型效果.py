import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


# 1. 读取评分数据

ratings = pd.read_csv(
    "ratings_Electronics (1).csv",
    names=["user_id", "product_id", "rating", "timestamp"]
)


# 2. 设置筛选阈值

MIN_USER_RATINGS = 5
MIN_PRODUCT_RATINGS = 60


# 3. 迭代筛选用户和商品

final_ratings = ratings.copy()

while True:

    before_len = len(final_ratings)

    # 保留评分次数 >= 5 的用户
    user_counts = final_ratings["user_id"].value_counts()

    valid_users = user_counts[
        user_counts >= MIN_USER_RATINGS
    ].index

    final_ratings = final_ratings[
        final_ratings["user_id"].isin(valid_users)
    ]

    # 保留被评分次数 >= 60 的商品
    product_counts = final_ratings["product_id"].value_counts()

    valid_products = product_counts[
        product_counts >= MIN_PRODUCT_RATINGS
    ].index

    final_ratings = final_ratings[
        final_ratings["product_id"].isin(valid_products)
    ]

    after_len = len(final_ratings)

    if after_len == before_len:
        break


print("最终筛选后的数据")

print("评分数：", len(final_ratings))
print("用户数：", final_ratings["user_id"].nunique())
print("商品数：", final_ratings["product_id"].nunique())


# 4. Leave-One-Out 拆分

train_parts = []
test_parts = []

for user_id, user_data in final_ratings.groupby("user_id"):

    # 找出该用户评分 >= 4 的商品
    high_rated = user_data[
        user_data["rating"] >= 4
    ]

    # 如果存在高评分商品
    if len(high_rated) > 0:

        # 随机挑 1 条高评分记录作为测试集
        test_row = high_rated.sample(
            n=1,
            random_state=42
        )

    else:

        # 如果没有 >= 4 分的商品
        # 就随机挑 1 条评分记录
        test_row = user_data.sample(
            n=1,
            random_state=42
        )

    # 剩余记录作为训练集
    train_rows = user_data.drop(
        test_row.index
    )

    train_parts.append(train_rows)
    test_parts.append(test_row)


# 5. 合并训练集和测试集

train_ratings = pd.concat(
    train_parts,
    ignore_index=True
)

test_ratings = pd.concat(
    test_parts,
    ignore_index=True
)


# 6. 检查拆分结果

print("\nTrain / Test Split")

print("训练集评分数：", len(train_ratings))
print("测试集评分数：", len(test_ratings))

print(
    "训练集用户数：",
    train_ratings["user_id"].nunique()
)

print(
    "测试集用户数：",
    test_ratings["user_id"].nunique()
)


print("\n测试集评分分布：")

print(
    test_ratings["rating"]
    .value_counts()
    .sort_index()
)


print("\n测试集前 10 条记录：")

print(
    test_ratings[
        [
            "user_id",
            "product_id",
            "rating"
        ]
    ].head(10)
)


# 7. 保留高评分测试商品

evaluation_test = test_ratings[
    test_ratings["rating"] >= 4
].copy()

print("\nEvaluation Test Set")

print(
    "用于评估的测试用户数：",
    evaluation_test["user_id"].nunique()
)

print(
    "用于评估的测试记录数：",
    len(evaluation_test)
)


# 8. 使用训练集建立 Product-User Matrix

train_user_codes, train_user_uniques = pd.factorize(
    train_ratings["user_id"]
)

train_product_codes, train_product_uniques = pd.factorize(
    train_ratings["product_id"]
)

train_product_user_matrix = csr_matrix(
    (
        train_ratings["rating"],
        (
            train_product_codes,
            train_user_codes
        )
    ),
    shape=(
        len(train_product_uniques),
        len(train_user_uniques)
    )
)

print("\nTraining Product-User Matrix")

print(
    "矩阵形状：",
    train_product_user_matrix.shape
)

print(
    "实际训练评分数量：",
    train_product_user_matrix.nnz
)


# 9. 只使用训练集计算商品相似度

train_item_similarity = cosine_similarity(
    train_product_user_matrix
)

print("\nTraining Similarity Matrix")

print(
    "相似度矩阵形状：",
    train_item_similarity.shape
)


# 10. 建立索引映射

user_to_index = {
    user_id: idx
    for idx, user_id in enumerate(train_user_uniques)
}

product_to_index = {
    product_id: idx
    for idx, product_id in enumerate(train_product_uniques)
}


# Product-User 转成 User-Product
train_user_item_matrix = (
    train_product_user_matrix
    .T
    .tocsr()
)


# 11. 检查测试商品是否存在于训练商品空间

evaluation_test = evaluation_test[
    evaluation_test["user_id"].isin(user_to_index)
    &
    evaluation_test["product_id"].isin(product_to_index)
].copy()

print("\n有效评估数据")

print(
    "有效评估用户数：",
    evaluation_test["user_id"].nunique()
)

print(
    "有效测试记录数：",
    len(evaluation_test)
)


# 12. Hit Rate@10 评估

TOP_K = 10
BATCH_SIZE = 500

hits = 0
total_users = 0


# 将测试数据转换成字典
test_product_map = dict(
    zip(
        evaluation_test["user_id"],
        evaluation_test["product_id"]
    )
)


evaluation_users = list(
    evaluation_test["user_id"]
)


for start in range(
    0,
    len(evaluation_users),
    BATCH_SIZE
):

    batch_users = evaluation_users[
        start:start + BATCH_SIZE
    ]

    # 找到这些用户在矩阵里的索引
    batch_user_indices = [
        user_to_index[user_id]
        for user_id in batch_users
    ]

    # 取出这一批用户的历史评分
    batch_matrix = train_user_item_matrix[
        batch_user_indices
    ]

    # 推荐分数
    #
    # User-Item Rating Matrix
    # ×
    # Item Similarity Matrix
    #
    # 得到每个用户对所有商品的推荐分数
    #
    scores = (
        batch_matrix
        @ train_item_similarity
    )


    # 排除用户已经评分过的商品

    for row_idx in range(
        len(batch_users)
    ):

        seen_items = (
            batch_matrix[row_idx]
            .indices
        )

        scores[
            row_idx,
            seen_items
        ] = float("-inf")


    # 找出每个用户 Top 10

    top_k_indices = (
        scores
        .argpartition(
            -TOP_K,
            axis=1
        )[:, -TOP_K:]
    )


    # 判断测试商品是否命中

    for row_idx, user_id in enumerate(
        batch_users
    ):

        test_product = (
            test_product_map[user_id]
        )

        test_product_index = (
            product_to_index[
                test_product
            ]
        )

        recommended_indices = (
            top_k_indices[row_idx]
        )

        if test_product_index in recommended_indices:
            hits += 1

        total_users += 1


# 13. 输出 Hit Rate@10

hit_rate = (
    hits / total_users
)

print("\nRecommendation Evaluation")

print(
    "测试用户数：",
    total_users
)

print(
    "Top 10 命中用户数：",
    hits
)

print(
    f"Hit Rate@10：{hit_rate:.4f}"
)

print(
    f"Hit Rate@10：{hit_rate * 100:.2f}%"
)