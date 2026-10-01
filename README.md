# Amazon 商业化数据分析项目

本项目围绕 Amazon 商品、用户评分和评论数据开展完整的数据分析流程，包括数据清洗与整合、探索性数据分析、评论文本分析、主题建模以及推荐系统构建与评估。

项目主要使用 Python、Pandas、NumPy、Matplotlib、NLTK、Scikit-learn 等工具，并使用 Tableau 构建综合可视化仪表盘。

---

## 项目背景与目标

本项目基于公开的 Amazon 电商数据开展商业化数据分析，主要目标是完成从数据获取、清洗、探索性分析、文本分析、推荐建模到商业洞察的完整分析流程。

项目重点关注以下问题：

- 不同商品类目的价格、折扣、评分和评价热度有何差异
- 高活跃用户与普通用户在评分行为上有何不同
- 用户评论中最常关注哪些产品属性和使用体验
- 是否存在“高评分但评论情感明显偏负面”的异常商品
- 如何利用用户评分行为构建基础推荐系统
- 不同推荐方法在相同实验设置下的效果有何差异

---

## 项目流程

本项目主要分为三个阶段。

### 第一阶段：数据工程与业务理解

主要完成：

- 原始数据读取与结构检查
- 缺失值、重复值和异常值处理
- 商品价格、折扣率、评分等字段格式转换
- 使用 IQR 方法识别价格异常值
- 商品数据与评分数据的关联测试
- 数据字典整理
- 业务指标可用性分析

Amazon 商品数据清洗后共保留：

- 1,247 条记录
- 1,136 个唯一商品

由于原始数据缺少真实销量、完整订单、曝光和点击等信息，因此无法严格计算 GMV、转化率和复购率。

因此后续分析主要使用以下可以可靠计算的指标：

- 商品价格
- 折扣率
- 商品评分
- 评价数量
- 用户评分行为
- 评论文本

---

## 第二阶段：探索性分析与可视化

### 1. 商品类目与价格分析

对主要商品类目的平均折后价、中位折后价以及价格分布进行比较。

主要结果：

- Home&Kitchen 整体价格水平最高
- OfficeProducts 整体价格水平最低
- Electronics 与 Computers&Accessories 存在较多高价商品

---

### 2. 商品类目多维度对比

进一步加入折扣率、平均评分和评价热度，对不同商品类目进行综合比较。

主要结果：

- Electronics 平均折扣率最高，为 57.34%
- Electronics 评分人数中位数最高，为 9,090
- OfficeProducts 平均评分最高，为 4.31
- Computers&Accessories 在价格、折扣和评分方面整体较为均衡

其中 `rating_count` 主要作为商品评价热度和用户关注度的代理指标，不代表真实销量。

---

### 3. 高活跃用户分析

根据用户评分商品数量和活跃时间对用户进行排序，并将排名前 10% 的用户定义为高活跃用户。

主要结果：

- 高活跃用户平均评价商品数：6.93
- 普通用户平均评价商品数：1.30
- 高活跃用户平均评分更高
- 高活跃用户好评率更高、差评率更低
- 高活跃用户平均活跃时间约为 1,041 天

这些结果表明，高活跃用户在平台中的互动频率和持续时间明显高于普通用户。

---

### 4. 总体评分分布

Amazon Electronics 评分数据整体明显集中在高评分区间。

主要结果：

- 1 星：11.52%
- 2 星：5.83%
- 3 星：8.09%
- 4 星：18.99%
- 5 星：55.56%

4 星和 5 星评分合计约占：

**74.55%**

总体平均评分约为：

**4.01**

评分中位数和众数均为：

**5**

---

### 5. 评论长度与商品评分关系

对商品评论文本长度与商品综合评分之间的关系进行分析。

结果显示：

- 高评分组的平均评论长度相对更高
- 但不同商品之间的评论长度差异较大
- 商品评分与评论字符数的相关系数约为 0.049
- 商品评分与评论单词数的相关系数约为 0.050

因此，评论长度和商品评分之间只存在非常弱的线性关系，不能认为“评论越长，评分越高”。

---

### 6. 评论高频词与词云

对评论文本进行清洗和词频统计，并生成词云。

较常出现的词包括：

- good
- quality
- cable
- price
- charging
- battery
- sound
- value

其中更具有商业解释价值的关注点包括：

- 产品质量
- 价格
- 充电性能
- 电池
- 音质
- 使用体验

词云主要用于观察评论中的关注点分布，不直接表示评论情感。

---

### 7. Tableau 综合仪表盘

使用 Tableau 整合多个分析结果，包括：

- Category Price
- User Comparison
- Rating Distribution
- Review Length
- Top Words

仪表盘用于集中展示商品、用户和评论三个维度的主要分析结果。

---

## 第三阶段：建模与深度洞察

### 1. 评论情感分析

使用 NLTK VADER 对商品级评论汇总文本进行情感分析。

情感分类标准：

- `compound >= 0.05`：Positive
- `compound <= -0.05`：Negative
- 其他：Neutral

分析结果：

- Positive：94.39%
- Negative：4.73%
- Neutral：0.88%

需要注意的是，数据中的 `review_content` 为商品级汇总评论文本，因此这里的分析单位是每个商品对应的评论文本集合，而不是单条独立评论。

---

### 2. 评分与评论情感异常分析

将商品综合评分与 VADER 情感得分结合分析。

多数高评分商品的评论文本情感同样为正，但仍存在部分异常商品，例如：

- 商品评分较高
- 但评论文本情感明显为负

这种“高评分 + 强负面文本”的情况可以作为商品体验异常监测信号，用于进一步排查产品质量、使用体验或评论内容。

---

### 3. LDA 主题建模

使用 `CountVectorizer` 将评论文本转换为词频矩阵，并使用 Latent Dirichlet Allocation（LDA）提取 5 个潜在主题。

识别出的主要主题包括：

1. Audio & Earphones
2. Smartwatch Features
3. Home Appliance Usage
4. General Product Experience
5. Charging & Cables

主题占比中：

- General Product Experience：54.53%
- Charging & Cables：24.38%
- Audio & Earphones：10.83%
- Smartwatch Features：8.50%
- Home Appliance Usage：1.76%

主题结果进一步表明，质量、价格、充电、电池和音质等是用户评论中较为普遍的关注点。

---

## 推荐系统

### 数据筛选

Amazon Electronics 原始用户—商品评分矩阵非常稀疏。

通过不同筛选阈值的对比实验，最终选择：

- 用户至少评价 5 个商品
- 商品至少获得 60 次评分

最终保留：

- 183,575 条评分记录
- 24,751 名用户
- 1,336 个商品
- 用户—商品矩阵密度：0.5552%

---

### Item-Based Collaborative Filtering

基于筛选后的 Product-User Matrix，使用 Cosine Similarity 计算商品之间的相似度。

模型的基本逻辑为：

1. 获取用户历史评分商品
2. 根据商品之间的相似度计算推荐分数
3. 排除用户已经评价过的商品
4. 为用户生成 Top-N 推荐结果

该方法不依赖商品文本属性，而是直接利用用户评分行为建立商品之间的关联。

---

## 推荐模型评估

使用 Leave-One-Out 方法进行推荐系统评估。

对于每名用户：

1. 优先随机选择一个评分不低于 4 分的商品作为测试商品
2. 其余评分作为训练数据
3. 仅使用训练数据重新构建模型
4. 判断测试商品是否出现在 Top-10 推荐列表中

最终有效评估用户数为：

**24,691**

Item-Based CF 的结果：

- Top-10 命中用户数：2,993
- Hit Rate@10：12.12%
- Precision@10：1.21%
- Recall@10：12.12%

在当前 Leave-One-Out 设置下，每名用户只有一个相关测试商品，因此 Recall@10 与 Hit Rate@10 数值相同。

---

## 四种推荐模型对比

为了比较不同方法的表现，本项目使用完全相同的训练集、测试集和 Top-10 评价设置，对四种模型进行实验。

| 模型 | Hit Rate@10 | Precision@10 | Recall@10 |
|---|---:|---:|---:|
| Popularity-Based | 5.95% | 0.59% | 5.95% |
| User-Based CF | 7.80% | 0.78% | 7.80% |
| Item-Based CF | 12.12% | 1.21% | 12.12% |
| Truncated SVD | 7.48% | 0.75% | 7.48% |

在当前数据和实验设置下：

**Item-Based Collaborative Filtering 的 Top-10 推荐表现最好。**

---

## 核心发现

本项目得到的主要结论包括：

- Electronics 的评价热度最高，同时折扣力度最大
- Home&Kitchen 的整体价格水平最高
- OfficeProducts 的平均评分最高
- 高活跃用户评价商品更多、活跃周期更长，同时整体评分行为更加正向
- 评论中的主要关注点包括质量、价格、充电、电池、音质和使用体验
- 部分商品存在“高评分但负面评论情感较强”的异常情况
- 在当前实验设置下，Item-Based CF 的 Top-10 推荐效果优于 Popularity-Based、User-Based CF 和 Truncated SVD

---

## 商业建议

基于分析结果，可以提出以下建议：

1. **Electronics**
   - 保持较高营销曝光
   - 同时关注折扣效率，避免过度依赖价格促销

2. **Home&Kitchen**
   - 重点关注高价格但低评价热度的商品
   - 优化价格、商品展示和促销策略

3. **高活跃用户**
   - 可以进行更精准的个性化推荐
   - 提供新品体验、定向活动和长期用户运营

4. **评论异常监测**
   - 可以结合商品评分与文本情感建立异常监控机制
   - 对“高评分 + 强负面评论”商品进行优先排查

5. **推荐系统**
   - 当前可以使用 Item-Based CF 作为基础推荐模型
   - 后续可加入商品内容、点击、购买等信息构建 Hybrid Recommendation System

---

## 项目目录

```text
Amazon-Commercial-Data-Analysis/
├── 01_Data_Engineering/
├── 02_EDA_Visualization/
├── 03_Modeling/
├── figures/
├── results/
├── tableau/
├── report/
├── presentation/
├── requirements.txt
├── .gitignore
└── README.md
```

### 文件夹说明

- `01_Data_Engineering/`
  - 数据清洗
  - 数据整合
  - 数据字典相关代码

- `02_EDA_Visualization/`
  - 探索性数据分析
  - 用户行为分析
  - 评论分析
  - 可视化与 Tableau 数据准备

- `03_Modeling/`
  - VADER 情感分析
  - LDA 主题建模
  - 推荐系统
  - 推荐模型评估与模型对比

- `figures/`
  - 项目主要可视化结果

- `results/`
  - 清洗后的数据和主要分析结果

- `tableau/`
  - Tableau 仪表盘相关数据

- `report/`
  - 项目最终分析报告

- `presentation/`
  - 项目总结演示文稿

---

## 数据来源

本项目使用两个公开的 Amazon 数据集，数据来源于 Kaggle。

### 1. Amazon Sales Dataset

本项目的商品分析、价格与折扣分析以及评论文本分析主要基于该数据集。

原始数据文件：

`amazon.csv`

数据主要包含：

- 商品 ID
- 商品名称
- 商品类目
- 原价
- 折后价
- 折扣率
- 商品评分
- 评价数量
- 评论文本

原始商品数据共包含：

- 1,465 条记录
- 16 个字段

数据来源：

https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset

---

### 2. Amazon Product Reviews

推荐系统和大规模用户评分分析主要使用 Amazon Electronics 用户评分数据。

原始文件：

`ratings_Electronics (1).csv`

主要字段包括：

- `user_id`
- `product_id`
- `rating`
- `timestamp`

该文件包含：

**7,824,482 条评分记录**

由于原始评分文件体积较大，因此未直接上传至本 GitHub 仓库。

数据来源：

https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews

---

### 数据使用说明

两个数据集中的商品 ID 覆盖范围存在较大差异，因此本项目没有强制将两套数据完全合并。

实际分析中：

- 商品、价格、折扣和评论文本分析主要使用 Amazon Sales Dataset
- 用户评分分析和推荐系统主要使用 Amazon Product Reviews

这种处理方式可以避免因为强制合并两个覆盖范围差异较大的数据源而造成大量有效数据损失。

---

## 使用的主要工具

- Python
- Pandas
- NumPy
- Matplotlib
- SciPy
- Scikit-learn
- NLTK
- WordCloud
- Tableau

Python 依赖记录在：

`requirements.txt`

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 运行说明

原始数据文件由于体积和版权等原因未全部上传至本仓库。

如需重新运行完整分析流程，请首先从上述数据来源下载对应数据。

主要原始文件包括：

```text
amazon.csv
ratings_Electronics (1).csv
```

本项目最初在统一的本地工作目录中完成开发。为了提高 GitHub 仓库的可读性，发布时将代码、结果和图表按照功能重新整理到了不同文件夹中。

因此，部分脚本中的相对文件路径仍基于原始开发环境。

如需重新运行相关脚本，请：

1. 下载所需原始数据
2. 根据脚本中的文件名准备数据
3. 根据当前目录结构调整相对路径或工作目录
4. 安装 `requirements.txt` 中的依赖
5. 按项目分析阶段依次运行相关脚本

---

## 项目局限性

本项目仍存在以下限制：

- 原始商品数据缺少真实销量、完整订单、曝光和点击数据
- 因此无法严格计算 GMV、转化率和复购率
- 两套 Amazon 数据的商品 ID 重合度较低
- 用户—商品评分矩阵具有较高稀疏性
- Amazon Electronics 评分数据属于历史数据，不能直接代表当前 Amazon 平台的实时用户行为
- 当前推荐系统主要基于显式评分数据
- 当前文本分析主要使用传统 NLP 方法，没有使用大型语言模型或更复杂的 Transformer 模型

---

## 后续优化方向

后续可以进一步尝试：

- 构建 Hybrid Recommendation System
- 将商品文本特征与 Collaborative Filtering 结合
- 引入点击、收藏、购买等隐式反馈数据
- 增加 NDCG、MAP 等推荐系统评价指标
- 使用更复杂的 Matrix Factorization 方法
- 对评论文本尝试 Transformer 或其他深度学习模型
- 建立自动化商品评分与负面情感异常监控机制
- 将分析流程进一步模块化并统一数据路径配置

---

## 作者

李重延
Arnold Li