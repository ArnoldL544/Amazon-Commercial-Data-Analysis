import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD


train_ratings = pd.read_csv(
    "comparison_train.csv"
)

evaluation_test = pd.read_csv(
    "comparison_test.csv"
)


TOP_K = 10
BATCH_SIZE = 500


user_codes, user_uniques = pd.factorize(
    train_ratings["user_id"]
)

product_codes, product_uniques = pd.factorize(
    train_ratings["product_id"]
)


user_item_matrix = csr_matrix(
    (
        train_ratings["rating"],
        (
            user_codes,
            product_codes
        )
    ),
    shape=(
        len(user_uniques),
        len(product_uniques)
    )
)


item_user_matrix = (
    user_item_matrix
    .T
    .tocsr()
)


user_to_index = {
    user_id: idx
    for idx, user_id in enumerate(user_uniques)
}

product_to_index = {
    product_id: idx
    for idx, product_id in enumerate(product_uniques)
}


evaluation_test = evaluation_test[
    evaluation_test["user_id"].isin(user_to_index)
    &
    evaluation_test["product_id"].isin(product_to_index)
].copy()


test_product_map = dict(
    zip(
        evaluation_test["user_id"],
        evaluation_test["product_id"]
    )
)


evaluation_users = list(
    evaluation_test["user_id"]
)


def calculate_metrics(hits, total_users):

    hit_rate = (
        hits / total_users
    )

    precision = (
        hits
        /
        (total_users * TOP_K)
    )

    recall = (
        hits / total_users
    )

    return (
        hit_rate,
        precision,
        recall
    )


results = []


print("\nPopularity-Based Recommendation")

product_popularity = (
    train_ratings["product_id"]
    .value_counts()
)

popular_products = list(
    product_popularity.index
)


user_history = (
    train_ratings
    .groupby("user_id")["product_id"]
    .apply(set)
    .to_dict()
)


hits = 0
total_users = 0


for user_id, test_product in test_product_map.items():

    seen_products = user_history.get(
        user_id,
        set()
    )

    recommendations = []

    for product_id in popular_products:

        if product_id not in seen_products:

            recommendations.append(
                product_id
            )

        if len(recommendations) == TOP_K:
            break

    if test_product in recommendations:
        hits += 1

    total_users += 1


hit_rate, precision, recall = calculate_metrics(
    hits,
    total_users
)


print("测试用户数：", total_users)
print("Top 10 命中用户数：", hits)
print(f"Hit Rate@10：{hit_rate * 100:.2f}%")
print(f"Precision@10：{precision * 100:.2f}%")
print(f"Recall@10：{recall * 100:.2f}%")


results.append({
    "Model": "Popularity-Based",
    "Hit Rate@10": hit_rate * 100,
    "Precision@10": precision * 100,
    "Recall@10": recall * 100
})


print("\nUser-Based Collaborative Filtering")


N_NEIGHBORS = 50


user_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=N_NEIGHBORS + 1
)


user_model.fit(
    user_item_matrix
)


hits = 0
total_users = 0


for start in range(
    0,
    len(evaluation_users),
    BATCH_SIZE
):

    batch_users = evaluation_users[
        start:start + BATCH_SIZE
    ]

    batch_user_indices = [
        user_to_index[user_id]
        for user_id in batch_users
    ]


    distances, neighbor_indices = (
        user_model.kneighbors(
            user_item_matrix[
                batch_user_indices
            ]
        )
    )


    for row_idx, user_id in enumerate(
        batch_users
    ):

        current_user_index = (
            user_to_index[user_id]
        )

        neighbors = (
            neighbor_indices[row_idx]
        )

        neighbor_distances = (
            distances[row_idx]
        )


        valid_neighbors = []
        similarities = []


        for neighbor_index, distance in zip(
            neighbors,
            neighbor_distances
        ):

            if neighbor_index == current_user_index:
                continue

            valid_neighbors.append(
                neighbor_index
            )

            similarities.append(
                1 - distance
            )


        valid_neighbors = np.array(
            valid_neighbors
        )

        similarities = np.array(
            similarities
        )


        neighbor_ratings = (
            user_item_matrix[
                valid_neighbors
            ]
        )


        scores = (
            neighbor_ratings.T
            @ similarities
        )


        scores = np.asarray(
            scores
        ).ravel()


        seen_items = (
            user_item_matrix[
                current_user_index
            ].indices
        )


        scores[
            seen_items
        ] = float("-inf")


        top_k_indices = (
            np.argpartition(
                scores,
                -TOP_K
            )[-TOP_K:]
        )


        test_product = (
            test_product_map[
                user_id
            ]
        )

        test_product_index = (
            product_to_index[
                test_product
            ]
        )


        if test_product_index in top_k_indices:
            hits += 1

        total_users += 1


hit_rate, precision, recall = calculate_metrics(
    hits,
    total_users
)


print("测试用户数：", total_users)
print("Top 10 命中用户数：", hits)
print(f"Hit Rate@10：{hit_rate * 100:.2f}%")
print(f"Precision@10：{precision * 100:.2f}%")
print(f"Recall@10：{recall * 100:.2f}%")


results.append({
    "Model": "User-Based CF",
    "Hit Rate@10": hit_rate * 100,
    "Precision@10": precision * 100,
    "Recall@10": recall * 100
})


print("\nItem-Based Collaborative Filtering")


item_similarity = cosine_similarity(
    item_user_matrix
)


hits = 0
total_users = 0


for start in range(
    0,
    len(evaluation_users),
    BATCH_SIZE
):

    batch_users = evaluation_users[
        start:start + BATCH_SIZE
    ]

    batch_user_indices = [
        user_to_index[user_id]
        for user_id in batch_users
    ]


    batch_matrix = user_item_matrix[
        batch_user_indices
    ]


    scores = (
        batch_matrix
        @ item_similarity
    )


    for row_idx in range(
        len(batch_users)
    ):

        seen_items = (
            batch_matrix[
                row_idx
            ].indices
        )

        scores[
            row_idx,
            seen_items
        ] = float("-inf")


    top_k_indices = (
        scores
        .argpartition(
            -TOP_K,
            axis=1
        )[:, -TOP_K:]
    )


    for row_idx, user_id in enumerate(
        batch_users
    ):

        test_product = (
            test_product_map[
                user_id
            ]
        )

        test_product_index = (
            product_to_index[
                test_product
            ]
        )


        if test_product_index in top_k_indices[row_idx]:
            hits += 1

        total_users += 1


hit_rate, precision, recall = calculate_metrics(
    hits,
    total_users
)


print("测试用户数：", total_users)
print("Top 10 命中用户数：", hits)
print(f"Hit Rate@10：{hit_rate * 100:.2f}%")
print(f"Precision@10：{precision * 100:.2f}%")
print(f"Recall@10：{recall * 100:.2f}%")


results.append({
    "Model": "Item-Based CF",
    "Hit Rate@10": hit_rate * 100,
    "Precision@10": precision * 100,
    "Recall@10": recall * 100
})


print("\nSVD Matrix Factorization")


N_COMPONENTS = 50


svd = TruncatedSVD(
    n_components=N_COMPONENTS,
    random_state=42
)


user_factors = svd.fit_transform(
    user_item_matrix
)


item_factors = (
    svd.components_.T
)


hits = 0
total_users = 0


for start in range(
    0,
    len(evaluation_users),
    BATCH_SIZE
):

    batch_users = evaluation_users[
        start:start + BATCH_SIZE
    ]

    batch_user_indices = [
        user_to_index[user_id]
        for user_id in batch_users
    ]


    batch_user_factors = (
        user_factors[
            batch_user_indices
        ]
    )


    scores = (
        batch_user_factors
        @ item_factors.T
    )


    batch_matrix = (
        user_item_matrix[
            batch_user_indices
        ]
    )


    for row_idx in range(
        len(batch_users)
    ):

        seen_items = (
            batch_matrix[
                row_idx
            ].indices
        )

        scores[
            row_idx,
            seen_items
        ] = float("-inf")


    top_k_indices = (
        scores
        .argpartition(
            -TOP_K,
            axis=1
        )[:, -TOP_K:]
    )


    for row_idx, user_id in enumerate(
        batch_users
    ):

        test_product = (
            test_product_map[
                user_id
            ]
        )

        test_product_index = (
            product_to_index[
                test_product
            ]
        )


        if test_product_index in top_k_indices[row_idx]:
            hits += 1

        total_users += 1


hit_rate, precision, recall = calculate_metrics(
    hits,
    total_users
)


print("测试用户数：", total_users)
print("Top 10 命中用户数：", hits)
print(f"Hit Rate@10：{hit_rate * 100:.2f}%")
print(f"Precision@10：{precision * 100:.2f}%")
print(f"Recall@10：{recall * 100:.2f}%")


results.append({
    "Model": "SVD",
    "Hit Rate@10": hit_rate * 100,
    "Precision@10": precision * 100,
    "Recall@10": recall * 100
})


results_df = pd.DataFrame(
    results
)


print("\nModel Comparison Results")

print(
    results_df.to_string(
        index=False
    )
)


results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)


x = np.arange(
    len(results_df)
)

width = 0.25


plt.figure(
    figsize=(11, 6)
)


plt.bar(
    x - width,
    results_df["Hit Rate@10"],
    width,
    label="Hit Rate@10"
)

plt.bar(
    x,
    results_df["Precision@10"],
    width,
    label="Precision@10"
)

plt.bar(
    x + width,
    results_df["Recall@10"],
    width,
    label="Recall@10"
)


plt.xticks(
    x,
    results_df["Model"],
    rotation=15
)

plt.ylabel(
    "Percentage (%)"
)

plt.xlabel(
    "Recommendation Model"
)

plt.title(
    "Recommendation Model Comparison"
)

plt.legend()

plt.tight_layout()


plt.savefig(
    "model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()