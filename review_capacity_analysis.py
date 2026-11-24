from pathlib import Path

import pandas as pd


# 从现有的开发集 OOF 阈值表读取结果
project_dir = Path(__file__).resolve().parent
table_path = (
    project_dir
    / "reports"
    / "tables"
    / "development_threshold_grid.csv"
)

threshold_table = pd.read_csv(table_path)

# 业务假设：最多复核 15% 的客户
max_review_rate = 0.15
false_negative_cost = 5
false_positive_cost = 1

threshold_table["scenario_cost"] = (
    false_negative_cost * threshold_table["false_negative"]
    + false_positive_cost * threshold_table["false_positive"]
)

# 筛选满足容量约束的阈值
feasible_thresholds = threshold_table.loc[
    threshold_table["predicted_high_risk_rate"] <= max_review_rate
].copy()

if feasible_thresholds.empty:
    raise ValueError("现有阈值表中没有满足复核容量约束的阈值。")

# 成本相同时，优先选择更高的阈值
selected = feasible_thresholds.sort_values(
    by=["scenario_cost", "threshold"],
    ascending=[True, False],
).iloc[0]

print(f"复核容量上限：{max_review_rate:.0%}")
print(f"开发集选定阈值：{selected['threshold']:.2f}")
print(
    f"开发集复核比例："
    f"{selected['predicted_high_risk_rate']:.2%}"
)
print(f"开发集违约召回率：{selected['recall']:.2%}")
print(f"开发集漏判违约数：{int(selected['false_negative'])}")
print(f"开发集示例成本：{int(selected['scenario_cost'])}")

# 使用开发集选好的阈值，评估留出测试集
selected_threshold = float(selected["threshold"])

test_path = (
    project_dir
    / "reports"
    / "tables"
    / "test_predictions.csv"
)

test_data = pd.read_csv(test_path)

actual_default = test_data["actual_default"].astype(int)
predicted_probability = test_data["predicted_probability"]

# 预测概率达到阈值的客户进入复核队列
needs_review = predicted_probability >= selected_threshold

# 进入复核队列且实际违约的客户
true_positive = int(
    (needs_review & (actual_default == 1)).sum()
)

# 未进入复核队列但实际违约的客户
false_negative = int(
    (~needs_review & (actual_default == 1)).sum()
)

# 进入复核队列但实际未违约的客户
false_positive = int(
    (needs_review & (actual_default == 0)).sum()
)

customer_count = len(test_data)
review_count = int(needs_review.sum())
default_count = int(actual_default.sum())

review_rate = review_count / customer_count
recall = true_positive / default_count if default_count else 0.0

scenario_cost = (
    false_negative_cost * false_negative
    + false_positive_cost * false_positive
)

print()
print("—— 留出测试集结果 ——")
print(f"固定阈值：{selected_threshold:.2f}")
print(f"测试集客户数：{customer_count}")
print(f"实际违约客户数：{default_count}")
print(f"复核人数：{review_count}")
print(f"复核比例：{review_rate:.2%}")
print(f"识别出的违约客户数：{true_positive}")
print(f"违约召回率：{recall:.2%}")
print(f"漏判违约数：{false_negative}")
print(f"误标正常客户数：{false_positive}")
print(f"示例成本：{scenario_cost}")

# 读取项目已有的默认阈值和成本阈值结果
original_comparison = pd.read_csv(
    project_dir / "reports" / "tables" / "threshold_comparison.csv"
)

comparison_columns = [
    "operating_point",
    "threshold",
    "high_risk_rate",
    "recall",
    "false_negative",
    "cost",
]

# 整理本次容量约束策略的测试集结果
capacity_result = pd.DataFrame([{
    "operating_point": f"Capacity {max_review_rate:.0%}",
    "threshold": selected_threshold,
    "high_risk_rate": review_rate,
    "recall": recall,
    "false_negative": false_negative,
    "cost": scenario_cost,
}])

comparison = pd.concat(
    [
        original_comparison[comparison_columns],
        capacity_result,
    ],
    ignore_index=True,
)

# 按阈值从高到低排列，并计算复核人数
comparison = comparison.sort_values(
    "threshold",
    ascending=False,
).reset_index(drop=True)

comparison["review_count"] = (
    comparison["high_risk_rate"] * customer_count
).round().astype(int)

comparison = comparison[[
    "operating_point",
    "threshold",
    "review_count",
    "high_risk_rate",
    "recall",
    "false_negative",
    "cost",
]]

# CSV 中保留数值，方便看板继续计算
output_path = (
    project_dir
    / "reports"
    / "tables"
    / "review_capacity_comparison.csv"
)

comparison.to_csv(output_path, index=False)

# 终端显示时，把比例转换成百分数
print()
print("—— 三种策略的测试集对照 ——")
print(comparison.to_string(
    index=False,
    formatters={
        "threshold": "{:.2f}".format,
        "high_risk_rate": "{:.2%}".format,
        "recall": "{:.2%}".format,
    },
))
print(f"\n对照表已保存到：{output_path}")