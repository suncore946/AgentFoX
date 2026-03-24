"""
难度分析模块
二分类取证检测的阈值无关、可复现难度度量系统

主要功能:
- 基于多模型聚合的样本难度度量
- 不确定性分解（认知vs数据不确定性）
- 噪声与域外样本检测
- IRT心理测量学难度分析
- 样本排序与筛选建议

使用示例:
    from src.difficult_analysis import DifficultyAnalyzer, analyze_difficulty_from_database

    # 从数据库进行分析
    analyzer = analyze_difficulty_from_database(
        dataset_name='AIGCDetect-testset',
        config='balanced'
    )

    # 获取最难的样本
    top_difficult = analyzer.get_top_difficult_samples(n=100)
"""

from .difficulty_calculator import DifficultyCalculator
from .noise_detector import NoiseDetector, SampleRiskType
from .irt_difficulty import IRTDifficultyEstimator
from ..difficult_generator import DifficultyGenerator, analyze_difficulty_from_database
