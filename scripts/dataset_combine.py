from pathlib import Path
import json
from collections import defaultdict

target_dir = Path("/data2/yuyangxin/Agent/resources/agent_test")
json_files = list(target_dir.rglob("*.json"))

tmp_data = {}
for file in json_files:
    method_name = file.parent.parent.name
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    tmp_data[method_name] = data

res = defaultdict(dict)
for method_name, data in tmp_data.items():
    for img_path, pred_info in data.items():
        res[img_path]["gt_label"] = pred_info["gt_label"]
        res[img_path][method_name] = pred_info["pred_label"]

output_file = target_dir / "combined_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=4)
