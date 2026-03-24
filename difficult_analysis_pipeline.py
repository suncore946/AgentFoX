from forensic_agent.difficult_analysis.difficult_generator import analyze_difficulty_from_database


def difficult_analysis():
    print("\n" + "=" * 60)
    print("示例2: 从数据库进行难度分析")
    print("=" * 60)

    try:
        # 从数据库加载并分析
        analyzer = analyze_difficulty_from_database(
            dataset_name="Genimage++",  # 分析特定数据集
            model_names=None,
            db_path="data/test_predictions.db",  # 使用测试数据库
            config="balanced",
            include_irt=True,
            max_samples=None,  # 限制样本数量以提高速度
            print_summary=True,
        )

        print("\n高价值训练样本统计:")
        recommendations = analyzer.results["sample_recommendations"]
        print(f"推荐训练样本: {len(recommendations['high_value_samples']['recommended_for_training'])}")
        print(f"稳健难样本: {len(recommendations['high_value_samples']['hard_valuable'])}")
        print(f"可疑样本: {len(recommendations['suspicious_samples']['all_suspicious'])}")

        return analyzer

    except Exception as e:
        print(f"数据库分析失败 (这是正常的，如果数据库不存在): {e}")
        return None


if __name__ == "__main__":
    difficult_analysis()
