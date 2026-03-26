# 读取目标目录下的所有csv文件
# csv的key为:method,dataset,sub_name,total_samples,overall_accuracy,overall_f1,overall_auc,real_samples,real_accuracy,real_f1,fake_samples,fake_accuracy,fake_f1
# 排除其他列, 仅关注method, dataset,sub_name, overall_accuracy

# 仅对一个dataset存在多个子sub_name的内容进行划分
# 根据dataset下不同的sub_name进行统计, 行为sub_name, 列为method, 值为overall_accuracy, 绘制二维表格
# 基于其数值的大小为每个表格单元格着色, 值越大颜色越深
# 每个dataset出一张图片

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from pathlib import Path
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

class DatasetAnalyzer:
    """数据集分析器类"""
    
    def __init__(self, data_dir="./data", output_dir="dataset_heatmaps"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.required_columns = ['method', 'dataset', 'sub_name', 'overall_accuracy']
        
    def read_csv_files(self):
        """读取目标目录下的所有csv文件，仅保留必要的列"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"目录 {self.data_dir} 不存在，请检查路径")
        
        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"在目录 {self.data_dir} 中未找到CSV文件")
        
        dataframes = []
        
        for file in csv_files:
            try:
                df = pd.read_csv(file, encoding='utf-8')
                
                # 检查必要列是否存在
                missing_columns = [col for col in self.required_columns if col not in df.columns]
                if missing_columns:
                    print(f"文件 {file.name} 缺少必要的列: {missing_columns}，跳过该文件")
                    continue
                
                # 仅保留必要的列并进行数据清洗
                df_filtered = df[self.required_columns].copy()
                df_filtered = df_filtered.dropna()
                
                # 确保overall_accuracy为数值类型
                df_filtered['overall_accuracy'] = pd.to_numeric(
                    df_filtered['overall_accuracy'], errors='coerce'
                )
                df_filtered = df_filtered.dropna()
                
                # 验证数据范围（accuracy应该在0-1或0-100之间）
                if not df_filtered['overall_accuracy'].between(0, 1).all():
                    if df_filtered['overall_accuracy'].between(0, 100).all():
                        # 如果是百分比形式，转换为小数
                        df_filtered['overall_accuracy'] = df_filtered['overall_accuracy'] / 100
                    else:
                        print(f"文件 {file.name} 的accuracy数值范围异常，跳过该文件")
                        continue
                
                if len(df_filtered) > 0:
                    dataframes.append(df_filtered)
                    print(f"成功读取文件: {file.name}，有效数据行数: {len(df_filtered)}")
                
            except UnicodeDecodeError:
                try:
                    # 尝试其他编码
                    df = pd.read_csv(file, encoding='gbk')
                    # 重复上述处理逻辑...
                except Exception as e:
                    print(f"读取文件 {file.name} 时出错: {e}")
            except Exception as e:
                print(f"读取文件 {file.name} 时出错: {e}")
        
        if not dataframes:
            raise ValueError("未找到任何有效的CSV文件")
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"总计读取有效数据: {len(combined_df)} 行")
        return combined_df

    def filter_datasets_with_multiple_subsets(self, df):
        """筛选出拥有多个sub_name的dataset"""
        print("开始筛选拥有多个sub_name的dataset...")
        
        # 计算每个dataset下的sub_name数量
        sub_name_counts = df.groupby('dataset')['sub_name'].nunique()
        
        print("所有dataset及其sub_name数量:")
        for dataset, count in sub_name_counts.items():
            print(f"  {dataset}: {count} 个sub_name")
        
        # 筛选出拥有多个sub_name的dataset
        datasets_with_multiple_subs = sub_name_counts[sub_name_counts > 1].index.tolist()
        
        if not datasets_with_multiple_subs:
            print("没有找到拥有多个sub_name的dataset")
            return pd.DataFrame()
        
        print(f"\n拥有多个sub_name的dataset: {datasets_with_multiple_subs}")
        
        filtered_df = df[df['dataset'].isin(datasets_with_multiple_subs)].copy()
        print(f"筛选后数据行数: {len(filtered_df)}")
        return filtered_df

    def create_pivot_table_for_dataset(self, df, dataset_name):
        """为特定dataset创建透视表"""
        dataset_data = df[df['dataset'] == dataset_name].copy()
        
        # 检查是否有重复的组合
        duplicates = dataset_data.groupby(['sub_name', 'method']).size()
        duplicates = duplicates[duplicates > 1]
        
        if not duplicates.empty:
            print(f"警告: 数据集 {dataset_name} 中存在重复的sub_name-method组合:")
            for (sub_name, method), count in duplicates.items():
                print(f"  {sub_name} + {method}: {count} 次")
        
        # 创建透视表
        pivot_table = dataset_data.pivot_table(
            index='sub_name', 
            columns='method', 
            values='overall_accuracy',
            aggfunc='mean'  # 如果有重复值，取平均
        )
        
        return pivot_table

    def create_heatmap_for_dataset(self, pivot_table, dataset_name):
        """为特定dataset创建热力图"""
        if pivot_table.empty:
            print(f"数据集 {dataset_name} 没有有效数据，跳过")
            return
        
        # 动态调整图片大小
        fig_width = max(12, len(pivot_table.columns) * 1.5)
        fig_height = max(8, len(pivot_table) * 1.0)
        
        plt.figure(figsize=(fig_width, fig_height))
        
        # 创建热力图
        mask = pivot_table.isnull()  # 为缺失值创建掩码
        
        heatmap = sns.heatmap(
            pivot_table, 
            annot=True,
            fmt='.4f',
            cmap='RdYlBu_r',  # 使用更好的颜色映射
            cbar_kws={'label': 'Overall Accuracy'},
            linewidths=0.5,
            linecolor='white',
            square=False,
            mask=mask,  # 隐藏缺失值
            vmin=pivot_table.min().min(),
            vmax=pivot_table.max().max()
        )
        
        plt.title(f'数据集 {dataset_name} - 各方法在不同子集上的准确率对比', 
                 fontsize=16, pad=20, fontweight='bold')
        plt.xlabel('方法 (Method)', fontsize=14, fontweight='bold')
        plt.ylabel('子数据集 (Sub Dataset)', fontsize=14, fontweight='bold')
        
        # 优化标签显示
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        
        plt.tight_layout()
        
        # 保存图片
        self.output_dir.mkdir(exist_ok=True)
        safe_dataset_name = self._safe_filename(dataset_name)
        save_path = self.output_dir / f"{safe_dataset_name}_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"热力图已保存到: {save_path}")

    def print_dataset_statistics(self, pivot_table, dataset_name):
        """打印特定dataset的统计信息"""
        print(f"\n{'='*20} 数据集 {dataset_name} 统计信息 {'='*20}")
        print(f"子数据集数量: {len(pivot_table)}")
        print(f"方法数量: {len(pivot_table.columns)}")
        print(f"数据完整度: {(1 - pivot_table.isnull().sum().sum() / pivot_table.size) * 100:.1f}%")
        
        # 各方法的平均accuracy
        print(f"\n各方法的平均准确率 (排序):")
        method_means = pivot_table.mean().sort_values(ascending=False)
        for i, (method, mean_acc) in enumerate(method_means.items(), 1):
            if not pd.isna(mean_acc):
                print(f"  {i:2d}. {method}: {mean_acc:.4f}")
        
        # 各sub_name的平均accuracy  
        print(f"\n各子数据集的平均准确率 (Top 5):")
        sub_name_means = pivot_table.mean(axis=1).sort_values(ascending=False)
        for i, (sub_name, mean_acc) in enumerate(sub_name_means.head().items(), 1):
            if not pd.isna(mean_acc):
                print(f"  {i}. {sub_name}: {mean_acc:.4f}")
        
        # 最佳表现组合
        max_value = pivot_table.max().max()
        if not pd.isna(max_value):
            max_pos = np.where(pivot_table.values == max_value)
            if len(max_pos[0]) > 0:
                best_sub_name = pivot_table.index[max_pos[0][0]]
                best_method = pivot_table.columns[max_pos[1][0]]
                print(f"\n🏆 最佳表现: {best_sub_name} + {best_method} = {max_value:.4f}")

    def _safe_filename(self, filename):
        """创建安全的文件名"""
        return filename.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_')

    def analyze(self):
        """主分析函数"""
        try:
            print("=" * 60)
            print("CSV数据分析工具 - 按Dataset分组分析")
            print("=" * 60)
            
            # 读取数据
            df = self.read_csv_files()
            print(f"\n✅ 数据读取完成！总数据行数: {len(df)}")
            
            # 显示数据预览
            print("\n📊 数据预览:")
            print(df.head())
            print(f"\n数据集概览:")
            print(f"  - 总数据集: {df['dataset'].nunique()}")
            print(f"  - 总方法: {df['method'].nunique()}")
            print(f"  - 总子数据集: {df['sub_name'].nunique()}")
            
            # 筛选数据
            filtered_df = self.filter_datasets_with_multiple_subsets(df)
            if filtered_df.empty:
                return
            
            # 获取符合条件的dataset
            datasets = filtered_df['dataset'].unique()
            print(f"\n🚀 开始为 {len(datasets)} 个数据集生成热力图...")
            
            # 处理每个dataset
            for i, dataset_name in enumerate(datasets, 1):
                print(f"\n📈 [{i}/{len(datasets)}] 处理数据集: {dataset_name}")
                
                # 创建透视表
                pivot_table = self.create_pivot_table_for_dataset(filtered_df, dataset_name)
                
                if not pivot_table.empty:
                    # 打印统计信息
                    self.print_dataset_statistics(pivot_table, dataset_name)
                    
                    # 创建热力图
                    self.create_heatmap_for_dataset(pivot_table, dataset_name)
                    
                    # 保存透视表
                    safe_dataset_name = self._safe_filename(dataset_name)
                    csv_output_path = self.output_dir / f"{safe_dataset_name}_pivot_table.csv"
                    pivot_table.to_csv(csv_output_path, encoding='utf-8-sig')
                    print(f"📊 透视表数据已保存到: {csv_output_path}")
                else:
                    print(f"❌ 数据集 {dataset_name} 生成的透视表为空")
            
            print(f"\n🎉 所有热力图已生成完成！输出目录: {self.output_dir}")
            
        except Exception as e:
            print(f"❌ 分析过程中出现错误: {e}")
            raise

def main():
    """主函数"""
    analyzer = DatasetAnalyzer(
        data_dir="./data",
        output_dir="dataset_heatmaps"
    )
    analyzer.analyze()

if __name__ == "__main__":
    main()