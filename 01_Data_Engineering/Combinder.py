import pandas as pd

# 读取主表
df_main = pd.read_csv('amazon_cleaned.csv')



chunk_size = 5000
merged_chunks = []

# 定义 4 列的列名
col_names = ['user_id', 'product_id', 'rating', 'timestamp']

for chunk in pd.read_csv('ratings_Electronics (1).csv', chunksize=chunk_size, header=None, names=col_names): 
    matched = chunk[chunk["product_id"].isin(df_main["product_id"])]
    merged_chunks.append(matched)
    
product_matched = pd.concat(merged_chunks, ignore_index = True)

df_final = pd.merge(df_main, product_matched, on = "product_id", how = "left", suffixes = ("_product", "_user"))

# 合并并保存
df_final.to_csv('amazon_wideTable.csv', index=False)
print(f"数据量为{len(df_final)} 条")