# 读取json文件

import json
from pathlib import Path

target_dir = Path("/data2/yuyangxin/Agent/outputs/agent_results_confuse_qwen3[qwen3_32b][open_calibration][open_clustering][open_semantic][open_expert]/final_output")
json_files = list(target_dir.glob("*.json"))


test = {}
# 读取每个json文件的内容
for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    image_path = data["image_path"]
    gt_label = data["gt_label"]
    
    test[image_path] = {
        "gt_label": gt_label  
    }
    
    model_res = data["expert_model"]
    for expert_name, expert_res in model_res.items():
        if int(expert_res > 0.5) == gt_label:
            expert_pred = 1
        else:
            expert_pred = 0
        test[image_path][f"{expert_name}_pred"] = expert_pred# 打印所有读取的内容
        
# 转为DataFrame
import pandas as pd
df = pd.DataFrame.from_dict(test, orient='index')

# expert_name总共有4个, 我想要找出4个expert都预测错误的样本
# 找出所有以 '_pred' 结尾的专家预测列
expert_pred_cols = [col for col in df.columns if col.endswith('_pred')]

# 计算每个样本有多少个专家预测错误（预测值为0）
df['wrong_expert_count'] = (df[expert_pred_cols] == 0).sum(axis=1)

# 筛选出所有专家都预测错误的样本
all_wrong_samples = df[df['wrong_expert_count'] == len(expert_pred_cols)]

print("="*50)
print(f"所有 {len(expert_pred_cols)} 个专家都预测错误的样本有 {len(all_wrong_samples)} 个:")
if not all_wrong_samples.empty:
    print(all_wrong_samples.index.tolist())
else:
    print("未找到所有专家都预测错误的样本。")
print("="*50)


# 总共有16种, 结合gt_label, 一共有32种情况, 每种情况给我展示2个样本
print("\n" + "="*50)
print("每种预测组合展示最多2个样本:")
print("="*50)

# 定义分组的列
grouping_cols = expert_pred_cols + ['gt_label']

# 按所有专家预测结果和gt_label进行分组，并取每组的前2个样本
samples_per_case = df.groupby(grouping_cols).head(2)

# 为了方便查看，我们按照分组列进行排序
sorted_samples = samples_per_case.sort_values(by=grouping_cols).set_index(grouping_cols, append=True)


# 打印结果
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200, 'display.max_colwidth', None):
    if not sorted_samples.empty:
        print(sorted_samples)

# 保存结果到CSV文件
output_csv_path = target_dir.parent / "samples_per_case.csv"
if not sorted_samples.empty:
    # 重置索引，以便将分组列和图像路径都作为普通列保存
    sorted_samples.reset_index().to_csv(output_csv_path, index=False)
    print(f"已将每种组合的最多2个样本信息保存到: {output_csv_path}")
else:
    print("没有可供保存的样本。")