from cfg import CONFIG
import json
import pandas as pd
from forensic_agent.data_operation.dataset_loader import load_project_data

# 加载项目数据
data, _ = load_project_data(**CONFIG["dataset"])

# 读取CSV文件
csv_data = "/data2/yuyangxin/Agent/outputs/dataset_sampling/GenImage/samples_by_label_with_clusters.csv"
df = pd.read_csv(csv_data)

# 检查 data 中同一 image_path 对应多个 calibration_prob 的情况并提示
dup_counts = data.groupby("image_path")["calibration_prob"].nunique()
conflict_count = (dup_counts > 1).sum()
if conflict_count > 0:
    print(f"警告: data 中有 {conflict_count} 个 image_path 对应多个不同的 calibration_prob，已保留第一个值")

# 生成去重的映射（保留第一个）
mapping_df = data[["image_path", "calibration_prob"]].drop_duplicates(subset="image_path", keep="first")

# 如果想用映射覆盖现有的 calibration_prob（推荐），先删除可能存在的列再合并
df = df.drop(columns=["calibration_prob"], errors="ignore")
df = df.merge(mapping_df, on="image_path", how="left")

# 检查是否有未匹配的数据
missing_count = df["calibration_prob"].isna().sum()
if missing_count > 0:
    print(f"警告: 有 {missing_count} 条记录未找到对应的calibration_prob值")

# 保存更新后的DataFrame（可选）
output_path = "/data2/yuyangxin/Agent/outputs/dataset_sampling/GenImage/samples_with_calibration.csv"
df.to_csv(output_path, index=False)

print(f"处理完成，共处理 {len(df)} 条记录")
print(f"成功匹配 {len(df) - missing_count} 条记录")
