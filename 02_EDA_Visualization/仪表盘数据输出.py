import pandas as pd

# 五个CSV文件及其对应的Excel工作表名称
files = {
    "category_price_summary.csv": "Category Price",
    "user_group_comparison.csv": "User Comparison",
    "rating_distribution.csv": "Rating Distribution",
    "review_length_rating_summary.csv": "Review Length",
    "top_20_review_words.csv": "Top Words"
}

# 创建整合后的Excel文件
with pd.ExcelWriter(
    "tableau_dashboard_data.xlsx",
    engine="openpyxl"
) as writer:

    for file_name, sheet_name in files.items():

        # 直接读取当前文件夹中的CSV
        data = pd.read_csv(file_name)

        # 写入Excel中的独立工作表
        data.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False
        )

        # 获取当前工作表
        worksheet = writer.sheets[sheet_name]

        # 固定第一行
        worksheet.freeze_panes = "A2"

        # 添加筛选功能
        worksheet.auto_filter.ref = worksheet.dimensions

        # 自动调整列宽
        for column_cells in worksheet.columns:

            column_letter = column_cells[0].column_letter

            max_length = max(
                len(str(cell.value))
                if cell.value is not None
                else 0
                for cell in column_cells
            )

            worksheet.column_dimensions[column_letter].width = min(
                max(max_length + 2, 12),
                35
            )

print("整合完成！")
print("已生成：tableau_dashboard_data.xlsx")