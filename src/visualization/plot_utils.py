"""
可视化工具模块
提供各种统计可视化功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import matplotlib
import logging

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)


class PlotUtils:
    """可视化工具类"""
    
    def __init__(self, style: str = 'seaborn', palette: str = 'Set2'):
        """
        初始化可视化工具
        
        Args:
            style: matplotlib样式
            palette: 颜色调色板
        """
        plt.style.use(style)
        self.palette = palette
        sns.set_palette(palette)
        logger.info(f"可视化工具初始化完成，使用样式: {style}")
    
    def plot_correlation_matrix(self, data: pd.DataFrame, 
                               method: str = 'pearson',
                               figsize: Tuple = (12, 10),
                               title: str = '相关性矩阵热力图',
                               save_path: Optional[str] = None):
        """
        绘制相关性矩阵热力图
        
        Args:
            data: 输入数据
            method: 相关性计算方法 ('pearson', 'spearman', 'kendall')
            figsize: 图形大小
            title: 图表标题
            save_path: 保存路径
        """
        # 计算相关性矩阵
        corr_matrix = data.corr(method=method)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制热力图
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                   square=True, linewidths=.5, annot=True, fmt='.2f',
                   cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"相关性矩阵已保存到: {save_path}")
        
        plt.show()
        
        return corr_matrix
    
    def plot_distribution(self, data: pd.DataFrame, 
                         columns: Optional[List] = None,
                         figsize: Tuple = (15, 10),
                         title: str = '变量分布分析',
                         save_path: Optional[str] = None):
        """
        绘制变量分布图
        
        Args:
            data: 输入数据
            columns: 要绘制的列，None表示所有数值列
            figsize: 图形大小
            title: 图表标题
            save_path: 保存路径
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = 3
        n_rows = int(np.ceil(len(columns) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # 直方图
            sns.histplot(data[col], kde=True, ax=ax, color='skyblue')
            
            # 添加统计信息
            mean_val = data[col].mean()
            median_val = data[col].median()
            
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'均值: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', alpha=0.7, label=f'中位数: {median_val:.2f}')
            
            ax.set_title(f'{col}分布', fontsize=12, fontweight='bold')
            ax.set_xlabel('')
            ax.legend(fontsize=9)
        
        # 隐藏多余的子图
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"分布图已保存到: {save_path}")
        
        plt.show()
    
    def plot_box_whisker(self, data: pd.DataFrame,
                        group_col: Optional[str] = None,
                        value_cols: Optional[List] = None,
                        figsize: Tuple = (14, 8),
                        title: str = '箱线图分析',
                        save_path: Optional[str] = None):
        """
        绘制箱线图
        
        Args:
            data: 输入数据
            group_col: 分组列
            value_cols: 值列
            figsize: 图形大小
            title: 图表标题
            save_path: 保存路径
        """
        if value_cols is None:
            value_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = 2
        n_rows = int(np.ceil(len(value_cols) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        
        axes = axes.flatten()
        
        for i, col in enumerate(value_cols):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            if group_col:
                # 分组箱线图
                groups = data[group_col].unique()
                group_data = [data[data[group_col] == group][col].dropna() 
                             for group in groups]
                
                bp = ax.boxplot(group_data, patch_artist=True)
                
                # 设置颜色
                colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax.set_xticklabels([str(g) for g in groups])
            else:
                # 单个箱线图
                ax.boxplot(data[col].dropna(), patch_artist=True)
            
            ax.set_title(f'{col}箱线图', fontsize=12, fontweight='bold')
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"箱线图已保存到: {save_path}")
        
        plt.show()
    
    def plot_time_series(self, data: pd.DataFrame,
                        time_col: str = '年份',
                        value_cols: List = None,
                        group_col: Optional[str] = None,
                        figsize: Tuple = (14, 8),
                        title: str = '时间序列分析',
                        save_path: Optional[str] = None):
        """
        绘制时间序列图
        
        Args:
            data: 输入数据
            time_col: 时间列
            value_cols: 值列
            group_col: 分组列
            figsize: 图形大小
            title: 图表标题
            save_path: 保存路径
        """
        if value_cols is None:
            value_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if time_col in value_cols:
                value_cols.remove(time_col)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, col in enumerate(value_cols[:4]):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            if group_col:
                # 分组绘制
                groups = data[group_col].unique()
                for group in groups:
                    group_data = data[data[group_col] == group]
                    ax.plot(group_data[time_col], group_data[col], 
                           marker='o', label=str(group))
                ax.legend(title=group_col)
            else:
                # 整体绘制
                ax.plot(data[time_col], data[col], marker='o', linewidth=2)
            
            ax.set_title(f'{col}时间趋势', fontsize=12, fontweight='bold')
            ax.set_xlabel(time_col)
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)
            
            # 添加趋势线
            if len(data) > 1:
                x_numeric = range(len(data))
                y = data[col].values
                z = np.polyfit(x_numeric, y, 1)
                p = np.poly1d(z)
                ax.plot(data[time_col], p(x_numeric), 'r--', alpha=0.7, label='趋势线')
                ax.legend()
        
        # 隐藏多余的子图
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"时间序列图已保存到: {save_path}")
        
        plt.show()


# 测试代码
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        '年份': list(range(2019, 2024)) * 20,
        '城市': ['城市' + str(i) for i in np.random.choice(5, n_samples)],
        'GDP': np.random.normal(1000, 200, n_samples),
        '数据流量': np.random.exponential(500, n_samples),
        '创新指数': np.random.uniform(50, 100, n_samples),
        '基础设施': np.random.normal(80, 15, n_samples)
    })
    
    # 创建可视化工具
    plot_utils = PlotUtils()
    
    # 测试相关性矩阵
    numeric_data = data[['GDP', '数据流量', '创新指数', '基础设施']]
    plot_utils.plot_correlation_matrix(numeric_data)
    
    # 测试分布图
    plot_utils.plot_distribution(data[['GDP', '数据流量', '创新指数']])
    
    print("✅ 可视化工具模块测试完成!")
