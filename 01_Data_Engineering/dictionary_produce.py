import pandas as pd

# 读取你清洗后的数据
df = pd.read_csv('amazon_cleaned.csv')

# 构建数据字典的原始数据
data_dict = {
    '字段名': df.columns,
    '数据类型': df.dtypes.astype(str),
    '业务含义/中文解释': [
        '商品的唯一识别码', 
        '商品名称', 
        '商品所属分类', 
        '商品促销后的折扣价格', 
        '商品原价', 
        '折扣百分比', 
        '商品综合评分', 
        '累计评价人数', 
        '商品详情描述', 
        '发表评论的用户ID', 
        '发表评论的用户名', 
        '单条评论的ID', 
        '评论的标题', 
        '评论的具体内容', 
        '商品的图片链接', 
        '商品详情页的链接'
    ],
    '示例数据': [df[col].iloc[0] if len(df) > 0 else '无' for col in df.columns]
}

# 转成 DataFrame
df_dict = pd.DataFrame(data_dict)

# === 核心修改：在列与列之间插入空列 ===
new_columns = []
for col in df_dict.columns:
    new_columns.append(col)   # 添加原来的数据列
    new_columns.append('')    # 添加空列
# 用重新排好的列名来重置表格
df_dict = df_dict.reindex(columns=new_columns)

# 保存为 Excel（空列也会被保留，形成间距）
df_dict.to_excel('Data_Dictionary.xlsx', index=False)
