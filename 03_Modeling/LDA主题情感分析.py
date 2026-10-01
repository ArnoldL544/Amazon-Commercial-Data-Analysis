import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


# 1. 读取数据
df = pd.read_csv("amazon_cleaned.csv")

print("数据行数：", len(df))


# 2. 提取评论文本
reviews = (
    df["review_content"]
    .dropna()
    .astype(str)
)

print("有效评论文本数量：", len(reviews))


# 3. 文本向量化
vectorizer = CountVectorizer(
    stop_words="english",
    lowercase=True,
    max_df=0.95,
    min_df=2,
    max_features=3000
)

review_matrix = vectorizer.fit_transform(reviews)

print("词汇数量：", len(vectorizer.get_feature_names_out()))


# 4. 建立 LDA 模型
lda = LatentDirichletAllocation(
    n_components=5,
    random_state=42,
    learning_method="batch",
    max_iter=20
)

lda.fit(review_matrix)

# 5. 计算每条评论属于各个主题的概率
topic_distribution = lda.transform(review_matrix)

# 每条评论选择概率最高的主题作为主导主题
dominant_topic = topic_distribution.argmax(axis=1)

# 保存到新的 DataFrame
topic_df = pd.DataFrame({
    "review_content": reviews.values,
    "dominant_topic": dominant_topic + 1
})


# 6. 统计每个主题的数量和占比
topic_counts = (
    topic_df["dominant_topic"]
    .value_counts()
    .sort_index()
)

topic_percent = (
    topic_counts / topic_counts.sum() * 100
).round(2)

print("\nTopic Distribution")

for topic_num in topic_counts.index:
    print(
        f"Topic {topic_num}: "
        f"{topic_counts[topic_num]} reviews "
        f"({topic_percent[topic_num]}%)"
    )


# 7. 输出每个主题的 Top 10 关键词
feature_names = vectorizer.get_feature_names_out()

for topic_index, topic in enumerate(lda.components_):

    top_word_indices = topic.argsort()[-10:][::-1]

    top_words = [
        feature_names[i]
        for i in top_word_indices
    ]

    print(
        f"\nTopic {topic_index + 1}:",
        ", ".join(top_words)
    )

# 8. 画柱状图
import matplotlib.pyplot as plt

topic_names = [
    "Audio & Earphones",
    "Smartwatch Features",
    "Home Appliance Usage",
    "General Product Experience",
    "Charging & Cables"
]

topic_percentages = [
    10.83,
    8.50,
    1.76,
    54.53,
    24.38
]

plt.figure(figsize=(10, 6))

plt.bar(
    topic_names,
    topic_percentages
)

plt.xlabel("Topic")
plt.ylabel("Percentage (%)")
plt.title("LDA Topic Distribution")

plt.xticks(rotation=20, ha="right")

plt.tight_layout()

plt.savefig(
    "lda_topic_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()