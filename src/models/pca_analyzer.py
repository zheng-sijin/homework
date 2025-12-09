"""
主成分分析(PCA)模块
实现PCA分析及多种评估方法
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional, Union
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PCAAnalyzer:
    """主成分分析器类"""
    
    def __init__(self, n_components: Optional[int] = None, 
                 variance_threshold: float = 0.85,
                 random_state: int = 42):
        """
        初始化PCA分析器
        
        Args:
            n_components: 主成分数量，None表示自动确定
            variance_threshold: 累计方差贡献率阈值
            random_state: 随机种子
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        self.pca = None
        self.scaler = StandardScaler()
        self.results = {}
        logger.info("PCA分析器初始化完成")
    
    def fit(self, X: pd.DataFrame, method: str = 'variance') -> 'PCAAnalyzer':
        """
        拟合PCA模型
        
        Args:
            X: 输入数据
            method: 主成分确定方法 ('variance', 'eigenvalue', 'scree')
            
        Returns:
            self
        """
        logger.info("开始拟合PCA模型")
        
        # 1. 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        self.X_scaled = X_scaled
        self.feature_names = X.columns.tolist()
        self.sample_names = X.index.tolist()
        
        # 2. 确定主成分数量
        if self.n_components is None:
            self.n_components = self._determine_n_components(X_scaled, method)
        
        logger.info(f"最终确定的主成分数量: {self.n_components}")
        
        # 3. 拟合PCA模型
        self.pca = PCA(n_components=self.n_components, 
                       random_state=self.random_state)
        self.pca.fit(X_scaled)
        
        # 4. 计算主成分得分
        self.components = self.pca.transform(X_scaled)
        
        # 5. 存储结果
        self._store_results(X)
        
        logger.info("PCA模型拟合完成")
        
        return self
    
    def _determine_n_components(self, X_scaled: np.ndarray, 
                               method: str = 'variance') -> int:
        """
        确定主成分数量
        
        Args:
            X_scaled: 标准化后的数据
            method: 确定方法 ('variance', 'eigenvalue', 'scree')
            
        Returns:
            主成分数量
        """
        logger.info(f"使用{method}方法确定主成分数量")
        
        # 先拟合一次PCA获取特征值
        pca_temp = PCA(random_state=self.random_state)
        pca_temp.fit(X_scaled)
        
        eigenvalues = pca_temp.explained_variance_
        variance_ratio = pca_temp.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratio)
        
        if method == 'variance':
            # 累计方差贡献率法
            n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
            logger.info(f"累计方差贡献率法确定主成分数: {n_components}")
            
        elif method == 'eigenvalue':
            # 特征值大于1法 (Kaiser准则)
            n_components = np.sum(eigenvalues > 1)
            logger.info(f"特征值大于1法确定主成分数: {n_components}")
            
        elif method == 'scree':
            # 碎石图拐点法
            n_components = self._scree_test(eigenvalues)
            logger.info(f"碎石图拐点法确定主成分数: {n_components}")
            
        else:
            n_components = min(X_scaled.shape[1], 10)
            logger.warning(f"未知方法，使用默认值: {n_components}")
        
        # 确保至少保留1个主成分
        n_components = max(1, n_components)
        
        return n_components
    
    def _scree_test(self, eigenvalues: np.ndarray) -> int:
        """碎石图拐点法"""
        # 计算二阶差分
        diff1 = np.diff(eigenvalues)
        diff2 = np.diff(diff1)
        
        # 寻找拐点（二阶差分最大负值）
        if len(diff2) > 0:
            elbow_point = np.argmin(diff2) + 2  # +2因为两次差分
        else:
            elbow_point = 1
        
        return elbow_point
    
    def _store_results(self, X: pd.DataFrame):
        """存储分析结果"""
        # 特征值和方差贡献率
        self.results['eigenvalues'] = self.pca.explained_variance_
        self.results['variance_ratio'] = self.pca.explained_variance_ratio_
        self.results['cumulative_variance'] = np.cumsum(self.pca.explained_variance_ratio_)
        
        # 因子载荷矩阵
        self.results['loadings'] = pd.DataFrame(
            self.pca.components_.T * np.sqrt(self.pca.explained_variance_),
            index=self.feature_names,
            columns=[f'PC{i+1}' for i in range(self.n_components)]
        )
        
        # 主成分得分
        self.results['scores'] = pd.DataFrame(
            self.components,
            index=self.sample_names,
            columns=[f'PC{i+1}' for i in range(self.n_components)]
        )
        
        # 变量共同度
        communalities = np.sum(self.results['loadings']**2, axis=1)
        self.results['communalities'] = pd.DataFrame(
            communalities,
            index=self.feature_names,
            columns=['Communality']
        )
    
    def plot_scree(self, save_path: Optional[str] = None):
        """
        绘制碎石图
        
        Args:
            save_path: 保存路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 碎石图（特征值）
        eigenvalues = self.results['eigenvalues']
        components = range(1, len(eigenvalues) + 1)
        
        ax1.plot(components, eigenvalues, 'bo-', linewidth=2, markersize=8)
        ax1.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='特征值=1')
        ax1.set_title('碎石图 (Scree Plot)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('主成分', fontsize=12)
        ax1.set_ylabel('特征值', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 标记选择的主成分
        if self.n_components is not None:
            ax1.axvline(x=self.n_components, color='g', linestyle='--', alpha=0.7, label=f'选择{self.n_components}个主成分')
        
        # 累计方差贡献率图
        cumulative_variance = self.results['cumulative_variance']
        
        ax2.plot(components, cumulative_variance, 'ro-', linewidth=2, markersize=8)
        ax2.axhline(y=self.variance_threshold, color='g', linestyle='--', alpha=0.7, label=f'阈值={self.variance_threshold}')
        ax2.set_title(f'累计方差贡献率', fontsize=14, fontweight='bold')
        ax2.set_xlabel('主成分', fontsize=12)
        ax2.set_ylabel('累计方差贡献率', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 标记选择的主成分
        if self.n_components is not None:
            ax2.axvline(x=self.n_components, color='g', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"碎石图已保存到: {save_path}")
        
        plt.show()
    
    def plot_biplot(self, pc1: int = 1, pc2: int = 2, 
                   save_path: Optional[str] = None):
        """
        绘制双标图
        
        Args:
            pc1: 第一个主成分
            pc2: 第二个主成分
            save_path: 保存路径
        """
        if pc1 > self.n_components or pc2 > self.n_components:
            raise ValueError("主成分索引超出范围")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 主成分得分
        scores = self.results['scores']
        x_scores = scores.iloc[:, pc1-1]
        y_scores = scores.iloc[:, pc2-1]
        
        # 因子载荷
        loadings = self.results['loadings']
        x_loadings = loadings.iloc[:, pc1-1]
        y_loadings = loadings.iloc[:, pc2-1]
        
        # 绘制样本点
        scatter = ax.scatter(x_scores, y_scores, alpha=0.7, s=100, 
                            edgecolors='k', linewidth=0.5)
        
        # 添加样本标签
        for i, label in enumerate(self.sample_names):
            ax.annotate(label, (x_scores.iloc[i], y_scores.iloc[i]), 
                       fontsize=10, alpha=0.7)
        
        # 绘制变量向量
        scale_factor = 5  # 调整向量长度
        for i, var in enumerate(loadings.index):
            ax.arrow(0, 0, 
                    x_loadings.iloc[i] * scale_factor, 
                    y_loadings.iloc[i] * scale_factor,
                    color='r', alpha=0.5, head_width=0.05)
            ax.text(x_loadings.iloc[i] * scale_factor * 1.1,
                   y_loadings.iloc[i] * scale_factor * 1.1,
                   var, color='r', fontsize=10, fontweight='bold')
        
        # 添加圆
        circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', alpha=0.3)
        ax.add_artist(circle)
        
        # 设置图形属性
        variance_ratio = self.results['variance_ratio']
        ax.set_xlabel(f'PC{pc1} ({variance_ratio[pc1-1]:.1%})', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC{pc2} ({variance_ratio[pc2-1]:.1%})', 
                     fontsize=12, fontweight='bold')
        ax.set_title('PCA双标图', fontsize=16, fontweight='bold')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # 设置坐标轴范围
        x_max = max(np.abs(x_scores).max(), np.abs(x_loadings * scale_factor).max()) * 1.2
        y_max = max(np.abs(y_scores).max(), np.abs(y_loadings * scale_factor).max()) * 1.2
        ax.set_xlim(-x_max, x_max)
        ax.set_ylim(-y_max, y_max)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"双标图已保存到: {save_path}")
        
        plt.show()
    
    def plot_loadings_heatmap(self, save_path: Optional[str] = None):
        """
        绘制因子载荷热力图
        
        Args:
            save_path: 保存路径
        """
        loadings = self.results['loadings']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 创建热力图
        sns.heatmap(loadings, annot=True, fmt='.2f', cmap='RdBu_r', 
                   center=0, ax=ax, cbar_kws={'label': '因子载荷'})
        
        ax.set_title('因子载荷矩阵热力图', fontsize=16, fontweight='bold')
        ax.set_xlabel('主成分', fontsize=12)
        ax.set_ylabel('变量', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"因子载荷热力图已保存到: {save_path}")
        
        plt.show()
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        获取结果摘要
        
        Returns:
            结果摘要DataFrame
        """
        summary_data = []
        
        for i in range(self.n_components):
            summary_data.append({
                '主成分': f'PC{i+1}',
                '特征值': self.results['eigenvalues'][i],
                '方差贡献率': self.results['variance_ratio'][i],
                '累计方差贡献率': self.results['cumulative_variance'][i]
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        return summary_df
    
    def interpret_components(self, threshold: float = 0.5) -> Dict:
        """
        解释主成分含义
        
        Args:
            threshold: 载荷阈值
            
        Returns:
            主成分解释字典
        """
        interpretations = {}
        loadings = self.results['loadings']
        
        for pc in loadings.columns:
            # 获取绝对值大于阈值的变量
            high_loadings = loadings[loadings[pc].abs() > threshold][pc]
            
            # 排序
            high_loadings = high_loadings.sort_values(key=abs, ascending=False)
            
            # 解释
            if len(high_loadings) > 0:
                # 正载荷变量
                positive_vars = high_loadings[high_loadings > 0].index.tolist()
                # 负载荷变量
                negative_vars = high_loadings[high_loadings < 0].index.tolist()
                
                interpretations[pc] = {
                    'positive_loading_vars': positive_vars,
                    'negative_loading_vars': negative_vars,
                    'interpretation': self._generate_interpretation(positive_vars, negative_vars)
                }
            else:
                interpretations[pc] = {
                    'positive_loading_vars': [],
                    'negative_loading_vars': [],
                    'interpretation': '无明显载荷变量，解释困难'
                }
        
        return interpretations
    
    def _generate_interpretation(self, positive_vars: List, 
                                negative_vars: List) -> str:
        """生成主成分解释文本"""
        if positive_vars and negative_vars:
            return f"综合反映了{len(positive_vars)}个正向指标和{len(negative_vars)}个负向指标"
        elif positive_vars:
            return f"主要反映正向指标：{', '.join(positive_vars[:3])}等"
        elif negative_vars:
            return f"主要反映负向指标：{', '.join(negative_vars[:3])}等"
        else:
            return "无明显模式"


# 测试代码
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    n_samples = 50
    n_features = 8
    
    # 生成相关数据
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'指标{i+1}' for i in range(n_features)],
        index=[f'样本{i+1}' for i in range(n_samples)]
    )
    
    # 添加一些相关性
    X['指标9'] = X['指标1'] * 0.7 + X['指标2'] * 0.3 + np.random.randn(n_samples) * 0.1
    X['指标10'] = X['指标3'] * 0.6 + X['指标4'] * 0.4 + np.random.randn(n_samples) * 0.1
    
    print("示例数据形状:", X.shape)
    
    # 创建PCA分析器
    pca_analyzer = PCAAnalyzer(variance_threshold=0.80)
    
    # 拟合PCA模型
    pca_analyzer.fit(X, method='variance')
    
    # 获取结果摘要
    summary = pca_analyzer.get_results_summary()
    print("\nPCA结果摘要:")
    print(summary)
    
    # 绘制图形
    pca_analyzer.plot_scree()
    pca_analyzer.plot_biplot()
    
    # 解释主成分
    interpretations = pca_analyzer.interpret_components(threshold=0.5)
    print("\n主成分解释:")
    for pc, info in interpretations.items():
        print(f"{pc}: {info['interpretation']}")
    
    print("\n✅ PCA分析模块测试完成!")
