from pathlib import Path


CONFIG = {
    # "test_dataset": "/data2/yuyangxin/Agent/outputs/dataset_sampling/GenImage-Val/samples_by_label.csv",
    # "test_dataset": "/data2/yuyangxin/Agent/outputs/dataset_sampling/sample_by_label/datasets_with_clusters_expert_model_analysis.csv",
    # "test_dataset": "/data2/yuyangxin/Agent/outputs/dataset_sampling/GenImage/samples_by_label.csv",
    "train_dataset": "/data2/yuyangxin/Agent/outputs/dataset_sampling/GenImage/sample_by_all/sampled_data.csv",
    # "test_dataset": "/data2/yuyangxin/Agent/outputs/x-fuse/test_dataset_CLIP_SRM_CFA_.csv",
    "test_dataset": "/data2/yuyangxin/Agent/outputs/genImage/result_scale1.4_merge.csv",
    # "val_dataset": "/data2/yuyangxin/Agent/outputs/dataset_sampling/GenImage-2M/samples_by_label_with_clusters_val.csv",
    # "test_dataset": "/data2/yuyangxin/Agent/outputs/dataset_sampling/WildRF_WIRA/sample_by_all/sampled_data.csv",
    # "test_dataset": "/data2/yuyangxin/Agent/outputs/dataset_sampling/WildRF_WIRA/sample_by_all/sampled_data.csv",
    #
    # "llm_pred_res": "/data2/yuyangxin/Agent/outputs/dataset_sampling/AIGCDetect-testset_AIGIBench_Chameleon_Community-Forensics-eval_WildRF_WIRA_GenImage-Val/sample_by_label/semantic/qwen3vl4B_evaluation_results.json",
    # "llm_pred_res": "/data2/yuyangxin/Agent/outputs/dataset_sampling/WildRF_WIRA/sample_by_all/semantic/semantic_with_qwen.json",
    # "train_dataset": "/data2/yuyangxin/Agent/outputs/x-fuse/train.csv",
    # "val_dataset": "/data2/yuyangxin/Agent/outputs/x-fuse/val.csv",
    # "test_dataset": "/data2/yuyangxin/Agent/outputs/x-fuse/test.csv",
    # "train_dataset": "/data2/yuyangxin/Agent/outputs/x-fuse/calibrated_train_results_GenImage-train.csv",
    # "val_dataset": "/data2/yuyangxin/Agent/outputs/x-fuse/calibrated_val_results_GenImage-val.csv",
    # "test_dataset": "/data2/yuyangxin/Agent/outputs/x-fuse/calibrated_test_results_x-fuse.csv",
    "llm": {
        "model": "gemini-3-pro-preview-thinking",
        "base_url": "https://api.apiyi.com",
        "api_key": "sk-KfgseAOy3hxsPmSgB36903DfA841406e9b47288f8bFc05Bd",
        "model_provider": "openai",
        "temperature": 0,
        "timeout": 60,
        "prompt_path": "/data2/yuyangxin/Agent/forensic_agent/configs/prompts/forensic_analysis_prompt.txt",
    },
    "ImageManager": {"max_width": 512, "max_height": 512},
    "feature": {
        "batch_size": 1000,
        "features": [
            "LaplacianVar",  # 清晰度/锐度
            "HF_LF_Ratio",  # 高低频比率
            "EdgeDensity",  # 边缘密度
            "Colorfulness",  # 颜色丰富度
            "Contrast_P95_P5",  # 对比度(P95-P5)
        ],
        "num_workers": 8,
        "use_gpu": True,
        "cache_dir": "./outputs/cache",
    },
    "clustering": {
        "force_clustering": False,
        "target_columns": [
            ["CLIP"],
            ["SRM"],
            ["CFA"],
            # ["CLIP", "CFA", "SRM"],
            # ["CLIP", "CFA"],
            # ["CFA", "SRM"],
            # ["LaplacianVar", "EdgeDensity", "HF_LF_Ratio"],
            # ["HF_LF_Ratio", "one_hot_result"],
            # ["ResNet-resnet50"],
            # ["CLIP", "CFA", "SRM"],
            # ["CFA", "SRM", "LaplacianVar", "EdgeDensity", "HF_LF_Ratio"],
            # ["LaplacianVar"],  # 清晰度
            # ["HF_LF_Ratio"],  # 频域信息
            # ["EdgeDensity"],  # 边缘信息
            # ["LaplacianVar", "HF_LF_Ratio"],  # 清晰度+频域信息
            # ["HF_LF_Ratio", "EdgeDensity"],  # 频域+边缘信息
            # ["LaplacianVar", "EdgeDensity"],  # 清晰度+边缘密度
            # ["LaplacianVar", "HF_LF_Ratio", "EdgeDensity", "Colorfulness", "Contrast_P95_P5"],
        ],
        "save_dir": "/data2/yuyangxin/Agent/outputs/genImage/cluster_models",
    },
    "profile": {
        "performance_threshold": 0.75,
        "significance_threshold": 0.05,
        "min_samples_for_analysis": 30,
        "llm": {
            "model": "claude-opus-4-1-20250805",
            # "base_url": "https://api.csun.site",
            # "api_key": "sk-OWLPeB1zJIiUw4tpvKO53e1LiovazisGY6K0BpDZhkq4AUlO",
            "base_url": "https://api.apiyi.com",
            "api_key": "sk-KfgseAOy3hxsPmSgB36903DfA841406e9b47288f8bFc05Bd",
            "temperature": 0.1,
            "timeout": 300,
            "model_provider": "OpenAI",
            "prompt_path": "./forensic_agent/configs/prompts/calibration_analysis_prompt.txt",
        },
        "model_profile": Path("./forensic_agent/configs/profiles/model_profiles.json"),
        "output_dir": Path("./outputs/clustering_profiles"),
    },
    "dataset": {
        # "datasets_root": "/home/yuyangxin/data/imdl-demo/aigc_model/kaiqing_eval",
        "db_dir": "./outputs/cache",
        "per_sample": 150,
        "model_names": [
            "DRCT",
            "RINE",
            "SPAI",
            "Patch_Shuffle",
            # "CLIPC2P",
            # "DeeCLIP",
        ],
        "dataset_names": [
            # "anime-test", # 没有伪造样本
            # "synthbuster", # 没有真实样本
            # "VLForgery", # 没有真实样本
            # "AIGCDetect-testset",
            # "AIGIBench",
            # "Chameleon",
            # "Community-Forensics-eval",
            # "WildRF",
            # "WIRA",
            # "GenImage-Val",
            "GenImage",
        ],
        "sampling_save_dir": "./outputs/dataset_sampling",
        # "dataset_names": ["Chameleon", "GenImage"],
        "force_reload": False,
    },
    "calibration": {
        "save_dir": Path("./outputs/calibration_results"),
        "llm": {
            "model": "claude-opus-4-1-20250805",
            # "base_url": "https://api.csun.site",
            # "api_key": "sk-OWLPeB1zJIiUw4tpvKO53e1LiovazisGY6K0BpDZhkq4AUlO",
            "base_url": "https://api.apiyi.com",
            "api_key": "sk-KfgseAOy3hxsPmSgB36903DfA841406e9b47288f8bFc05Bd",
            "temperature": 0.1,
            "presence_penalty": 0,
            "timeout": 300,
            "model_provider": "OpenAI",
        },
        "calibration_analysis_prompt_path": "./forensic_agent/configs/prompts/calibration_analysis_prompt.txt",
        "model_profile_prompt_path": "./forensic_agent/configs/prompts/model_profile_prompt.txt",
    },
}
