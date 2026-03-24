from pathlib import Path
import re
import pandas as pd
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm
from forensic_agent.core.forensic_llm import ForensicLLM
from forensic_agent.processor.detection_processor import DetectionProcessor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from forensic_agent.data_operation.dataset_loader import load_project_data
from forensic_agent.core.core_exceptions import JSONParsingError
from cfg import CONFIG


def load_data() -> pd.DataFrame:
    """加载数据并进行基本验证"""
    try:
        if "test_dataset" in CONFIG:
            data = pd.read_csv(CONFIG["test_dataset"])
            return data
        else:
            print("正在从数据库加载数据...")
            data, _ = load_project_data(**CONFIG["dataset"])

            if data is None or data.empty:
                raise ValueError("加载的数据为空")

            if "image_path" not in data.columns:
                raise ValueError("数据中缺少 image_path 列")

            print(f"✅ 加载了 {len(data)} 条预测记录")
            return data

    except Exception as e:
        print(f"❌ 加载数据时出错: {e}")
        raise


def parse_prediction_result(response):
    """从响应中解析预测结果"""
    pred_label = str(response.get("pred_label")).strip().lower()
    if pred_label in ["0", 0, "real", "真实"]:
        return 0  # 真实
    elif pred_label in ["1", 1, "fake", "伪造", "fake image", "manipulated", "AI生成", "AI生成图像"]:
        return 1  # 伪造
    else:
        raise JSONParsingError(f"无法解析的 pred_label: {pred_label}, 请确保返回值为 0 或 1")


def parse_error_result(response: str):
    """从响应中解析预测结果"""
    # 尝试正则表达式从response中提取pred_label后的值, 如果无法提取则返回-1
    response = response.strip()

    # 转换为小写以便匹配
    response_lower = response.lower()

    # 定义匹配模式，按优先级排序
    patterns = [
        # 匹配 pred_label: 值 或 pred_label=值
        r'pred_label\s*[:=]\s*["\']?(\w+)["\']?'
    ]

    for pattern in patterns:
        match = re.search(pattern, response_lower)
        if match:
            value = match.group(1)
            # 根据匹配的值判断标签
            if value in ["0", "real", "true", "真实"]:
                return 0  # 真实
            elif value in ["1", "fake", "false", "伪造", "ai生成"]:
                return 1  # 伪造
            # 如果是纯数字，直接转换
            elif value.isdigit():
                label = int(value)
                return label if label in [0, 1] else -1
    # 如果都无法匹配，返回-1表示无法确定
    return -1


def process_single_image(image_path, processor: DetectionProcessor, gt_label):
    """处理单个图像的函数，返回结果字典"""
    # 验证图像文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    try:
        response = processor.run(image_path=image_path)
        response["pred_label"] = parse_prediction_result(response)  # 解析预测结果
        response["gt_label"] = gt_label  # 添加真实标签到响应中
        return image_path, {"status": "success", **response}
    except JSONParsingError as e:
        return image_path, {
            "status": "success",
            "pred_label": parse_error_result(str(e)),
            "gt_label": gt_label,
            "parsing_error": str(e),
        }


def process_expert_model_by_group(data_group: pd.DataFrame, model_name: str, target_column: str = "pred_prob"):
    """
    处理单个专家模型组的数据

    Args:
        data_group: 属于某个模型的数据组
        model_name: 模型名称

    Returns:
        dict: 包含该模型的预测结果
    """
    results = {}

    print(f"📊 处理专家模型: {model_name} ({len(data_group)} 个样本)")

    for idx, row in tqdm(data_group.iterrows(), total=len(data_group), desc=f"处理 {model_name}", unit="img"):
        image_path = row["image_path"]
        gt_label = row["gt_label"]

        # 验证图像文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        # 专家模型预测：概率>0.5为正例(fake=1)，否则为负例(real=0)
        expert_pred = 1 if row[target_column] >= 0.5 else 0
        results[image_path] = {
            "status": "success",
            "pred_label": expert_pred,
            "gt_label": gt_label,
        }

    return results


def calculate_metrics(y_true, y_pred):
    """
    计算分类性能指标

    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表

    Returns:
        dict: 包含各种性能指标的字典
    """
    # 统计失败预测数量
    failed_predictions = sum(1 for pred in y_pred if pred is None)

    # 过滤并转换预测结果
    valid_pairs = []
    for gt, pred in zip(y_true, y_pred):
        if pred is None:
            continue
        elif pred == -1:
            # 将-1视为与真实标签相反的预测结果
            valid_pairs.append((gt, 1 - gt))
        else:
            valid_pairs.append((gt, pred))

    if not valid_pairs:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "valid_predictions": 0,
            "total_predictions": len(y_pred),
            "failed_predictions": failed_predictions,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    y_true_valid, y_pred_valid = zip(*valid_pairs)

    # 计算指标
    accuracy = accuracy_score(y_true_valid, y_pred_valid)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_valid, y_pred_valid, average="macro")
    cm = confusion_matrix(y_true_valid, y_pred_valid)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "valid_predictions": len(valid_pairs),
        "total_predictions": len(y_pred),
        "failed_predictions": failed_predictions,
    }


def calculate_model_metrics(results: dict):
    """
    计算单个模型的性能指标

    Args:
        results: 处理结果

    Returns:
        dict: 该模型的性能指标
    """
    y_true = []
    y_pred = []

    for res in results.values():
        y_true.append(res["gt_label"])
        y_pred.append(res["pred_label"]) if res["status"] == "success" else y_pred.append(-1)

    # 计算整体指标
    metrics = calculate_metrics(y_true, y_pred)

    # 分别计算Real(0)和Fake(1)样本的指标
    y_0_true, y_0_pred = [], []
    y_1_true, y_1_pred = [], []

    for res in results.values():
        if res["gt_label"] == 0:  # Real样本
            y_0_true.append(res["gt_label"])
            if res["status"] == "success":
                y_0_pred.append(res["pred_label"])
            else:
                y_0_pred.append(-1)
        else:  # Fake样本 (gt_label == 1)
            y_1_true.append(res["gt_label"])
            if res["status"] == "success":
                y_1_pred.append(res["pred_label"])
            else:
                y_1_pred.append(-1)

    metrics_0 = calculate_metrics(y_0_true, y_0_pred)  # Real样本指标
    metrics_1 = calculate_metrics(y_1_true, y_1_pred)  # Fake样本指标

    # 统计成功和失败数量
    successful = sum(1 for res in results.values() if res["status"] == "success" and res["pred_label"] != -1)
    failed = len(results) - successful

    return {
        "All_Metrics": metrics,
        "Real_Metrics": metrics_0,  # Real样本(label=0)的指标
        "Fake_Metrics": metrics_1,  # Fake样本(label=1)的指标
        "sample_count": len(results),
        "successful": successful,
        "failed": failed,
        "success_rate": successful / len(results) if len(results) > 0 else 0,
    }


def save_model_results(model_name: str, data_group: pd.DataFrame, results: dict, metrics: dict, config: dict, output_dir: Path):
    """
    保存单个模型的结果到JSON文件

    Args:
        model_name: 模型名称
        data_group: 该模型的数据
        results: 处理结果
        metrics: 性能指标
        config: 配置信息
        output_dir: 输出目录
    """
    # 过滤出属于该模型的结果
    model_results = {}
    for _, row in data_group.iterrows():
        image_path = row["image_path"]
        if image_path in results:
            model_results[image_path] = results[image_path]

    final_results = {
        "metadata": {
            "model_name": model_name,
            "processor_type": "expert_model",
            "total_images": len(data_group),
            "successful": metrics["successful"],
            "failed": metrics["failed"],
            "success_rate": metrics["success_rate"],
            "config": config,
            "processing_time": datetime.now().isoformat(),
        },
        "metrics": metrics["metrics"],
        "results": model_results,
    }

    # 构建文件名
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    filename = f"baseline_expert_{safe_model_name}.json"
    output_path = output_dir / filename

    try:
        os.makedirs(output_dir, exist_ok=True)

        # 备份现有文件
        if os.path.exists(output_path):
            backup_path = f"{output_path}.backup"
            os.rename(output_path, backup_path)
            print(f"📁 已备份 {model_name} 现有文件到: {backup_path}")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)

        print(f"💾 {model_name} 结果已保存到: {output_path}")

    except Exception as e:
        print(f"❌ 保存 {model_name} 结果时出错: {e}")
        raise


def save_results_to_json(results, output_path="./outputs/baseline/forensic_results.json"):
    """保存结果到JSON文件，增加错误处理"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 备份现有文件
        if os.path.exists(output_path):
            backup_path = f"{output_path}.backup"
            os.rename(output_path, backup_path)
            print(f"📁 已备份现有文件到: {backup_path}")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"💾 结果已保存到: {output_path}")

    except Exception as e:
        print(f"❌ 保存结果时出错: {e}")
        raise


def validate_config(config):
    """验证配置参数"""
    required_keys = ["llm", "dataset", "ImageManager"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"配置中缺少必需的键: {key}")

    # 验证LLM配置
    llm_required = ["model", "base_url"]
    for key in llm_required:
        if key not in config["llm"]:
            raise ValueError(f"LLM配置中缺少必需的键: {key}")


def main(max_workers=8, output_dir="./outputs", limit=None, test_expert_model=False, is_debug=False):
    output_dir = Path(output_dir) / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 验证配置
        validate_config(CONFIG)

        # 加载和处理数据
        data: pd.DataFrame = load_data()

        # 根据模式验证数据列并处理数据
        if test_expert_model:
            # 专家模型模式：按model_name分组处理
            if "pred_prob" not in data.columns:
                raise ValueError("专家模型模式时，数据中必须包含 pred_prob 列")
            if "model_name" not in data.columns:
                raise ValueError("专家模型模式时，数据中必须包含 model_name 列")

            print(f"🔬 启用专家模型模式，将按 model_name 分组处理")
            print(f"📊 发现模型: {data['model_name'].unique().tolist()}")

            # 专家模型模式不需要对image_path去重，因为不同模型可能对同一图像有不同预测
            data_processed = data.copy()

        else:
            # LLM模式：对image_path去重，不分组
            print(f"🤖 启用LLM模式")
            print("🔄 LLM模式：对image_path进行去重...")
            # 去重处理 - 保留每个image_path的第一条记录
            data_processed = data.drop_duplicates(subset=["image_path"], keep="first")
            print(f"去重前: {len(data)} 条记录，去重后: {len(data_processed)} 条记录")

        # 过滤存在的图像文件
        print("🔍 验证图像文件...")
        data_processed.loc[:, "file_exists"] = data_processed["image_path"].apply(os.path.exists)
        existing_data = data_processed[data_processed["file_exists"] == True].copy()
        if existing_data.empty:
            raise ValueError("没有找到有效的图像文件")

        if len(existing_data) != len(data_processed):
            missing_count = len(data_processed) - len(existing_data)
            print(f"⚠️  发现 {missing_count} 个不存在的图像文件")

        # 应用限制
        if limit and limit > 0:
            if test_expert_model:
                # 专家模型模式：按模型分组应用限制
                limited_data = []
                for model_name, group in existing_data.groupby("model_name"):
                    model_limit = min(limit // len(existing_data["model_name"].unique()), len(group))
                    if model_limit > 0:
                        limited_data.append(group.head(model_limit))

                if limited_data:
                    existing_data = pd.concat(limited_data, ignore_index=True)
                    print(f"🔢 专家模型模式按模型分组限制处理，总共 {len(existing_data)} 个图像")
                else:
                    print("⚠️ 限制数量太小，无法为每个模型分配样本")
            else:
                # LLM模式：直接限制
                existing_data = existing_data.head(limit)
                print(f"🔢 LLM模式限制处理前 {limit} 个图像")

        print(f"🚀 准备处理 {len(existing_data)} 个图像")

        # 根据模式处理
        if test_expert_model:
            # 专家模型模式：按模型分组处理
            # 打印每个模型的样本数量
            for model_name, group in existing_data.groupby("model_name"):
                print(f"  📋 {model_name}: {len(group)} 个样本")

            basic_metrics = {}
            calibrated_metrics = {}
            for model_name, data_group in existing_data.groupby("model_name"):
                print(f"\n{'='*60}")
                print(f"🔄 开始处理专家模型: {model_name}")
                print(f"{'='*60}")

                # 处理该模型的数据
                model_results = process_expert_model_by_group(data_group, model_name, target_column="pred_prob")
                # 计算该模型的性能指标
                model_metrics = calculate_model_metrics(model_results)
                # 存储到全局结果中
                basic_metrics[model_name] = model_metrics

                # 处理该模型的数据
                model_results = process_expert_model_by_group(data_group, model_name, target_column="calibration_prob")
                # 计算该模型的性能指标
                model_metrics = calculate_model_metrics(model_results)
                # 存储到全局结果中
                calibrated_metrics[model_name] = model_metrics

            # 保存专家模型汇总结果
            summary_results = {
                "metadata": {
                    "processor_type": "expert_model",
                    "total_models": len(basic_metrics),
                    "model_names": list(basic_metrics.keys()),
                    "total_images": len(existing_data),
                    "processing_time": datetime.now().isoformat(),
                },
                "basic_metrics": basic_metrics,
                "calibrated_metrics": calibrated_metrics,
            }

            summary_path = output_dir / f"baseline_expert_summary.json"
            save_results_to_json(summary_results, summary_path)

            # 打印最终统计
            print("\n" + "=" * 60)
            print("📊 专家模型处理完成统计:")
            print("=" * 60)

            total_successful = sum(metrics["successful"] for metrics in basic_metrics.values())
            total_failed = sum(metrics["failed"] for metrics in basic_metrics.values())

            for model_name, metrics in basic_metrics.items():
                print(f"\n📋 {model_name}:")
                print(f"  ✅ 成功: {metrics['successful']}")
                print(f"  ❌ 失败: {metrics['failed']}")
                print(f"  📊 成功率: {metrics['success_rate']:.4f}")
                print(f"  🎯 准确率: {metrics['all_metrics']['accuracy']:.4f}")
                print(f"  🎯 F1分数: {metrics['all_metrics']['f1_score']:.4f}")

            print(f"\n🎯 总体统计:")
            print(f"✅ 总成功: {total_successful}")
            print(f"❌ 总失败: {total_failed}")
            print(f"📈 总计: {len(existing_data)}")
            print(f"💾 汇总结果已保存到: {summary_path}")

            return summary_results

        else:
            # LLM模式：不分组，直接处理所有图像
            print("🔧 初始化LLM和处理器...")
            forensic_llm = ForensicLLM(CONFIG["llm"])
            complex_reasoning_processor = DetectionProcessor(CONFIG, llm=forensic_llm.llm)

            # 存储结果的字典
            results = {}

            if is_debug:
                print("🐞 调试模式：使用单线程顺序处理（不并发）")
                with tqdm(total=len(existing_data), desc="调试：处理LLM图像", unit="img") as pbar:
                    for idx, row in existing_data.iterrows():
                        image_path = row["image_path"]
                        gt_label = row["gt_label"]
                        try:
                            path, result = process_single_image(image_path, complex_reasoning_processor, gt_label)
                            results[path] = result
                        except TimeoutError:
                            print(f"\n⏰ 任务超时 {image_path}")
                            results[image_path] = {"status": "timeout", "pred_label": -1, "error": "处理超时"}
                        except Exception as e:
                            print(f"\n❌ 任务执行异常 {image_path}: {e}")
                            results[image_path] = {"status": "error", "pred_label": parse_error_result(str(e)), "error": str(e)}
                        finally:
                            pbar.update(1)

                        break
            else:
                # 多线程并发处理
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # 提交所有任务
                    future_to_image = {}

                    # 直接遍历DataFrame的行
                    for idx, row in existing_data.iterrows():
                        image_path = row["image_path"]
                        gt_label = row["gt_label"]
                        future = executor.submit(process_single_image, image_path, complex_reasoning_processor, gt_label)
                        future_to_image[future] = image_path

                    # 使用tqdm显示进度条
                    with tqdm(total=len(future_to_image), desc="处理LLM图像", unit="img") as pbar:
                        try:
                            for future in as_completed(future_to_image):
                                image_path = future_to_image[future]
                                try:
                                    # 为每个任务设置单独的超时（例如60秒）
                                    path, result = future.result(timeout=60)
                                    results[path] = result
                                except TimeoutError:
                                    print(f"\n⏰ 任务超时 {image_path}")
                                    results[image_path] = {"status": "timeout", "pred_label": -1, "error": "处理超时"}
                                    future.cancel()  # 尝试取消任务
                                except Exception as e:
                                    print(f"\n❌ 任务执行异常 {image_path}: {e}")
                                    results[image_path] = {"status": "error", "pred_label": parse_error_result(str(e)), "error": str(e)}
                                finally:
                                    pbar.update(1)
                        except TimeoutError:
                            print(f"\n⏰ 整体处理超时，正在取消剩余任务...")
                            # 取消所有未完成的任务
                            for future in future_to_image:
                                if not future.done():
                                    future.cancel()
                                    image_path = future_to_image[future]
                                    results[image_path] = {"status": "cancelled", "pred_label": -1, "error": "任务被取消"}
                                    pbar.update(1)
                        except KeyboardInterrupt:
                            print(f"\n🛑 用户中断，正在取消所有任务...")
                            # 取消所有未完成的任务
                            for future in future_to_image:
                                if not future.done():
                                    future.cancel()
                            raise

            # 计算LLM模式的性能指标
            print("\n📊 计算LLM性能指标...")

            y_true = []
            y_pred = []
            successful = 0
            failed = 0

            for image_path, result in results.items():
                y_true.append(result.get("gt_label", 0))
                if result["status"] == "success":
                    if result["pred_label"] == -1:
                        y_pred.append(-1)
                        failed += 1
                    else:
                        y_pred.append(result["pred_label"])
                        successful += 1
                else:
                    y_pred.append(-1)
                    failed += 1

            # 计算指标
            metrics = calculate_metrics(y_true, y_pred)

            # 构建最终结果
            final_results = {
                "metadata": {
                    "processor_type": "llm_model",
                    "model": CONFIG["llm"]["model"],
                    "total_images": len(existing_data),
                    "successful": successful,
                    "failed": failed,
                    "max_workers": max_workers,
                    "processing_time": datetime.now().isoformat(),
                },
                "metrics": metrics,
                "results": results,
            }

            print("💾 保存LLM结果...")
            # 保存LLM结果
            output_path = output_dir / f"baseline_llm_{CONFIG['llm']['model']}.json"
            save_results_to_json(final_results, output_path)

            # 打印处理统计
            print("\n" + "=" * 50)
            print("📊 LLM模式处理完成统计:")
            print(f"✅ 成功: {successful}")
            print(f"❌ 失败: {failed}")
            print(f"📈 总计: {len(existing_data)}")

            print(f"\n🎯 LLM性能指标:")
            print(f"准确率: {metrics['accuracy']:.4f}")
            print(f"精确率: {metrics['precision']:.4f}")
            print(f"召回率: {metrics['recall']:.4f}")
            print(f"F1分数: {metrics['f1_score']:.4f}")
            print(f"有效预测: {metrics['valid_predictions']}/{metrics['total_predictions']}")
            print(f"💾 结果已保存到: {output_path}")
            print("=" * 50)

            return final_results

    except Exception as e:
        print(f"❌ 主函数执行出错: {e}")
        raise


if __name__ == "__main__":
    # # 专家模型模式 - 按model_name分组处理，不去重
    # main(test_expert_model=True, output_dir="outputs")

    # LLM模式 - 不分组，对image_path去重
    main(test_expert_model=False, max_workers=32, output_dir="outputs", is_debug=False)
