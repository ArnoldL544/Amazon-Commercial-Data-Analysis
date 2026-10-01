
import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv("amazon.csv")


#处理缺失值

# 去掉价格里的特殊符号
df['discounted_price'] = df['discounted_price'].astype(str).str.replace(r'[$₹,]', '', regex=True)
df['actual_price'] = df['actual_price'].astype(str).str.replace(r'[$₹,]', '', regex=True)
df['rating_count'] = df['rating_count'].astype(str).str.replace(r'[$₹,]', '', regex=True)


# 把需要做数学运算的值全部转成数字类型
cols_to_convert = ['discounted_price', 'actual_price', 'rating', 'rating_count']
for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# 将缺失值补全或删除
df['rating_count'] = df['rating_count'].astype("Int64")
df=df.dropna(subset=['rating'])




# 处理异常值

# IQR测范围
# 算出 25% 和 75% 分位数的差值
Q1 = df['discounted_price'].quantile(0.25)
Q3 = df['discounted_price'].quantile(0.75)
IQR = Q3 - Q1

# 找出正常范围
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"价格正常范围是：{lower_bound} ~ {upper_bound}")

# 只保留正常范围的数据
df = df[(df['discounted_price'] >= lower_bound) & (df['discounted_price'] <= upper_bound)]




# 把清洗完的数据保存成新的文件
df.to_csv('amazon_cleaned.csv', index=False)