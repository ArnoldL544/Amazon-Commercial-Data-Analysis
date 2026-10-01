import pandas as pd


# 1. 读取评分数据

ratings = pd.read_csv(
    "ratings_Electronics (1).csv",
    names=["user_id", "product_id", "rating", "timestamp"]
)

print("原始数据")

print("总评分数：", len(ratings))
print("用户数：", ratings["user_id"].nunique())
print("商品数：", ratings["product_id"].nunique())


# 2. 定义阈值测试函数

def test_filter_thresholds(
    ratings,
    min_user_ratings,
    min_product_ratings
):

    data = ratings.copy()

    # 迭代筛选，直到没有更多记录被删除
    while True:

        before_len = len(data)

        # 筛选用户

        user_counts = data["user_id"].value_counts()

        valid_users = user_counts[
            user_counts >= min_user_ratings
        ].index

        data = data[
            data["user_id"].isin(valid_users)
        ]


        # 筛选商品

        product_counts = data["product_id"].value_counts()

        valid_products = product_counts[
            product_counts >= min_product_ratings
        ].index

        data = data[
            data["product_id"].isin(valid_products)
        ]


        after_len = len(data)

        # 如果这一轮没有继续删除数据，则结束
        if after_len == before_len:
            break


    # 3. 统计筛选后的规模

    rating_num = len(data)
    user_num = data["user_id"].nunique()
    product_num = data["product_id"].nunique()


    # 防止用户数或商品数为 0
    if user_num == 0 or product_num == 0:
        density = 0

    else:
        density = (
            rating_num
            / (user_num * product_num)
            * 100
        )


    # 4. 输出结果

    print(
        f"User >= {min_user_ratings}, "
        f"Product >= {min_product_ratings}"
    )

    print("评分数：", rating_num)
    print("用户数：", user_num)
    print("商品数：", product_num)
    print(f"矩阵密度：{density:.4f}%")
    print(f"矩阵稀疏度：{100 - density:.4f}%")

    print()


# 5. 测试不同阈值

test_filter_thresholds(
    ratings,
    min_user_ratings=5,
    min_product_ratings=50
)

test_filter_thresholds(
    ratings,
    min_user_ratings=5,
    min_product_ratings=60
)

test_filter_thresholds(
    ratings,
    min_user_ratings=5,
    min_product_ratings=70
)

test_filter_thresholds(
    ratings,
    min_user_ratings=5,
    min_product_ratings=80
)

test_filter_thresholds(
    ratings,
    min_user_ratings=6,
    min_product_ratings=50
)

test_filter_thresholds(
    ratings,
    min_user_ratings=7,
    min_product_ratings=50
)

test_filter_thresholds(
    ratings,
    min_user_ratings=6,
    min_product_ratings=60
)