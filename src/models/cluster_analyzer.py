"""
聚类分析模块
实现K-Means聚类算法及评估方法
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ClusterAnalyzer:
    """聚类分析器类"""
    
    def __init__(self, max_clusters: int = 10, random_state: int = 42):
        """
        初始化聚类分析器
        
        Args:
            max_clusters: 最大聚类数
            random_state: 随机种子
        """
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.results = {}
        logger.info("聚类分析器初始化完成")
    
    def determine_optimal_clusters(self, X: np.ndarray) -> Dict:
        """
        确定最优聚类数
        
        Args:
            X: 输入数据
            
        Returns:
            最优聚类数结果字典
        """
        logger.info("确定最优聚类数")
        
        results = {}
        n_clusters_range = range(2, min(self.max_clusters + 1, len(X) + 1))
        
        # 肘部法则（Inertia）
        inertia_values = []
        silhouette_scores = []
        calinski_scores = []
        davies_scores = []
        
        for k in n_clusters_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            
            inertia_values.append(kmeans.inertia_)
            
            if len(np.unique(labels)) > 1:
                silhouette_scores.append(silhouette_score(X, labels))
                calinski_scores.append(calinski_harabasz_score(X, labels))
                davies_scores.append(davies_bouldin_score(X, labels))
            else:
                silhouette_scores.append(0)
                calinski_scores.append(0)
                davies_scores.append(np.inf)
        
        # 肘部法则：寻找拐点
        diff1 = np.diff(inertia_values)
        diff2 = np.diff(diff1)
        
        if len(diff2) > 0:
            elbow_point = np.argmin(diff2) + 2  # +2因为两次差分
            optimal_k_elbow = list(n_clusters_range)[elbow_point]
        else:
            optimal_k_elbow = 2
        
        # 轮廓系数：最大值
        optimal_k_silhouette = list(n_clusters_range)[np.argmax(silhouette_scores)]
        
        # Calinski-Harabasz：最大值
        optimal_k_calinski = list(n_clusters_range)[np.argmax(calinski_scores)]
        
        # Davies-Bouldin：最小值
        optimal_k_davies = list(n_clusters_range)[np.argmin(davies_scores)]
        
        # 综合推荐（投票）
        recommendations = [optimal_k_elbow, optimal_k_silhouette, optimal_k_calinski, optimal_k_davies]
        optimal_k = max(set(recommendations), key=recommendations.count)
        
        results = {
            'inertia': inertia_values,
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'davies_scores': davies_scores,
            'optimal_k_elbow': optimal_k_elbow,
            'optimal_k_silhouette': optimal_k_silhouette,
            'optimal_k_calinski': optimal_k_calinski,
            'optimal_k_davies': optimal_k_davies,
            'recommended_k': optimal_k,
            'n_clusters_range': list(n_clusters_range)
        }
        
        logger.info(f"推荐聚类数: {optimal_k}")
        
        return results
    
    def apply_kmeans(self, X: pd.DataFrame, 
                    n_clusters: Optional[int] = None,
                    **kwargs) -> Dict:
        """
        应用K-Means聚类
        
        Args:
            X: 输入数据
            n_clusters: 聚类数量，None表示自动确定
            **kwargs: KMeans算法参数
            
        Returns:
            聚类结果字典
        """
        logger.info("应用K-Means聚类算法")
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        self.X_scaled = X_scaled
        self.feature_names = X.columns.tolist()
        self.sample_names = X.index.tolist()
        
        # 确定聚类数量
        if n_clusters is None:
            optimal_results = self.determine_optimal_clusters(X_scaled)
            n_clusters = optimal_results['recommended_k']
            logger.info(f"自动确定最优聚类数: {n_clusters}")
        
        # 应用K-Means
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
            **kwargs
        )
        
        labels = kmeans.fit_predict(X_scaled)
        centers = kmeans.cluster_centers_
        
        # 存储结果
        result = {
            'algorithm': 'kmeans',
            'labels': labels,
            'centers': centers,
            'inertia': kmeans.inertia_,
            'n_iter': kmeans.n_iter_,
            'n_clusters': n_clusters
        }
        
        # 评估聚类质量
        if len(np.unique(labels)) > 1:
            result['evaluation'] = {
                'silhouette_score': silhouette_score(X_scaled, labels),
                'calinski_harabasz_score': calinski_harabasz_score(X_scaled, labels),
                'davies_bouldin_score': davies_bouldin_score(X_scaled, labels)
            }
        
        self.results['kmeans'] = result
        self.current_labels = labels
        
        logger.info(f"K-Means聚类完成，共{n_clusters}个聚类")
        
        return result
    
    def plot_cluster_evaluation(self, X: np.ndarray, 
                               save_path: Optional[str] = None):
        """
        绘制聚类评估图
        
        Args:
            X: 输入数据
            save_path: 保存路径
        """
        # 计算不同聚类数的评估指标
        evaluation_results = self.determine_optimal_clusters(X)
        
        n_clusters_range = evaluation_results['n_clusters_range']
        silhouette_scores = evaluation_results['silhouette_scores']
        calinski_scores = evaluation_results['calinski_scores']
        davies_scores = evaluation_results['davies_scores']
        inertia_values = evaluation_results['inertia']
        
        # 绘制图形
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 肘部法则（Inertia）
        axes[0, 0].plot(n_clusters_range, inertia_values, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_title('肘部法则 (Inertia)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('聚类数', fontsize=12)
        axes[0, 0].set_ylabel('Inertia', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 标记推荐点
        if 'recommended_k' in evaluation_results:
            rec_k = evaluation_results['recommended_k']
            rec_idx = n_clusters_range.index(rec_k)
            axes[0, 0].plot(rec_k, inertia_values[rec_idx], 'ro', markersize=10, label=f'推荐: k={rec_k}')
            axes[0, 0].legend()
        
        # 2. 轮廓系数
        axes[0, 1].plot(n_clusters_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        axes[0, 1].set_title('轮廓系数', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('聚类数', fontsize=12)
        axes[0, 1].set_ylabel('轮廓系数', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Calinski-Harabasz指数
        axes[1, 0].plot(n_clusters_range, calinski_scores, 'go-', linewidth=2, markersize=8)
        axes[1, 0].set_title('Calinski-Harabasz指数', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('聚类数', fontsize=12)
        axes[1, 0].set_ylabel('Calinski-Harabasz指数', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Davies-Bouldin指数
        axes[1, 1].plot(n_clusters_range, davies_scores, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_title('Davies-Bouldin指数', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('聚类数', fontsize=12)
        axes[1, 1].set_ylabel('Davies-Bouldin指数', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('聚类数评估指标', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"聚类评估图已保存到: {save_path}")
        
        plt.show()
    
    def plot_clusters(self, X: pd.DataFrame, 
                     labels: np.ndarray,
                     features: Optional[List] = None,
                     save_path: Optional[str] = None):
        """
        绘制聚类结果
        
        Args:
            X: 输入数据
            labels: 聚类标签
            features: 要可视化的特征，None表示使用前两个特征
            save_path: 保存路径
        """
        if features is None:
            features = X.columns[:2].tolist()
        
        if len(features) < 2:
            logger.warning("需要至少2个特征进行可视化")
            return
        
        # 获取唯一标签
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 为每个聚类设置颜色
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            cluster_label = f'聚类{label}'
            
            ax.scatter(X.iloc[mask, X.columns.get_loc(features[0])],
                      X.iloc[mask, X.columns.get_loc(features[1])],
                      c=[colors[i]], s=100, alpha=0.7,
                      edgecolors='k', linewidth=0.5, label=cluster_label)
        
        # 添加聚类中心（如果是K-Means）
        if 'kmeans' in self.results and 'centers' in self.results['kmeans']:
            centers = self.results['kmeans']['centers']
            # 注意：centers是标准化后的，需要反标准化显示
            centers_original = self.scaler.inverse_transform(centers)
            
            # 创建临时的centers DataFrame用于获取列索引
            temp_df = pd.DataFrame(centers_original, columns=self.feature_names)
            
            ax.scatter(temp_df[features[0]], temp_df[features[1]],
                      c='red', s=300, marker='*', label='聚类中心')
        
        ax.set_xlabel(features[0], fontsize=12, fontweight='bold')
        ax.set_ylabel(features[1], fontsize=12, fontweight='bold')
        ax.set_title(f'K-Means聚类结果可视化 (共{n_clusters}个聚类)', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"聚类图已保存到: {save_path}")
        
        plt.show()
    
    def get_cluster_summary(self, X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """
        获取聚类摘要
        
        Args:
            X: 输入数据
            labels: 聚类标签
            
        Returns:
            聚类摘要DataFrame
        """
        summary_data = []
        
        for label in np.unique(labels):
            mask = labels == label
            cluster_data = X[mask]
            
            summary_data.append({
                '聚类编号': label,
                '样本数': np.sum(mask),
                '占比': f"{np.sum(mask) / len(X):.1%}",
                '特征均值': cluster_data.mean().tolist(),
                '特征标准差': cluster_data.std().tolist()
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        return summary_df


# 测试代码
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    n_samples = 100
    n_features = 4
    
    # 生成3个明显的聚类
    cluster1 = np.random.randn(n_samples//3, n_features) + np.array([0, 0, 0, 0])
    cluster2 = np.random.randn(n_samples//3, n_features) + np.array([5, 5, 5, 5])
    cluster3 = np.random.randn(n_samples//3, n_features) + np.array([10, 0, 0, 10])
    
    X_data = np.vstack([cluster1, cluster2, cluster3])
    X = pd.DataFrame(X_data, 
                     columns=[f'特征{i+1}' for i in range(n_features)],
                     index=[f'样本{i+1}' for i in range(n_samples)])
    
    print("示例数据形状:", X.shape)
    
    # 创建聚类分析器
    cluster_analyzer = ClusterAnalyzer(max_clusters=8)
    
    # 确定最优聚类数
    optimal_results = cluster_analyzer.determine_optimal_clusters(X.values)
    print(f"推荐聚类数: {optimal_results['recommended_k']}")
    
    # 应用K-Means聚类
    kmeans_result = cluster_analyzer.apply_kmeans(X, n_clusters=optimal_results['recommended_k'])
    
    # 评估聚类质量
    if 'evaluation' in kmeans_result:
        evaluation = kmeans_result['evaluation']
        print(f"\n聚类质量评估:")
        print(f"轮廓系数: {evaluation['silhouette_score']:.3f}")
        print(f"Calinski-Harabasz指数: {evaluation['calinski_harabasz_score']:.1f}")
        print(f"Davies-Bouldin指数: {evaluation['davies_bouldin_score']:.3f}")
    
    # 绘制评估图
    cluster_analyzer.plot_cluster_evaluation(X.values)
    
    # 绘制聚类图
    cluster_analyzer.plot_clusters(X, kmeans_result['labels'])
    
    # 获取聚类摘要
    summary = cluster_analyzer.get_cluster_summary(X, kmeans_result['labels'])
    print(f"\n聚类摘要:")
    print(summary[['聚类编号', '样本数', '占比']])
    
    print("\n✅ 聚类分析模块测试完成!")
