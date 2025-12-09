"""
数据加载模块
负责加载和整合各类数据源
"""

import pandas as pd
import numpy as np
import os
import yaml
import json
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器类"""
    
    def __init__(self, project_root: str = None):
        """
        初始化数据加载器
        
        Args:
            project_root: 项目根目录，如果为None则尝试自动检测
        """
        # 确定项目根目录
        if project_root is None:
            # 尝试从当前文件位置推断
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..')
        
        self.project_root = project_root
        
        try:
            # 加载配置文件
            config_path = os.path.join(project_root, 'config', 'config.yaml')
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
            self.config = {
                'data': {
                    'raw_data_path': 'data/raw/',
                    'od_matrix_file': 'od_matrix.csv',
                    'main_data_file': 'main_data_advanced.csv',
                    'cities': ["香港", "澳门", "广州", "深圳", "珠海", "佛山", "惠州", "东莞", "中山", "江门", "肇庆"]
                }
            }
        
        # 设置数据目录
        self.data_dir = os.path.join(project_root, 'data', 'raw')
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info(f"数据加载器初始化完成，数据目录: {self.data_dir}")
    
    def load_od_matrix(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        加载OD矩阵数据
        
        Args:
            file_path: 文件路径，如果为None则使用配置中的路径
            
        Returns:
            OD矩阵DataFrame
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, self.config['data']['od_matrix_file'])
        
        logger.info(f"正在加载OD矩阵数据: {file_path}")
        
        try:
            # 尝试不同的编码方式读取CSV文件
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='gbk')
            
            logger.info(f"OD矩阵数据加载成功，形状: {df.shape}")
            logger.info(f"数据列: {df.columns.tolist()}")
            logger.info(f"年份范围: {df['年份'].min()} - {df['年份'].max()}")
            logger.info(f"城市数量: {df['起点城市'].nunique()}")
            
            return df
            
        except Exception as e:
            logger.error(f"加载OD矩阵数据失败: {e}")
            raise
    
    def load_main_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        加载城市年度数据
        
        Args:
            file_path: 文件路径，如果为None则使用配置中的路径
            
        Returns:
            城市年度数据DataFrame
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, self.config['data']['main_data_file'])
        
        logger.info(f"正在加载城市年度数据: {file_path}")
        
        try:
            # 尝试不同的编码方式读取CSV文件
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='gbk')
            
            logger.info(f"城市年度数据加载成功，形状: {df.shape}")
            logger.info(f"数据列: {df.columns.tolist()}")
            logger.info(f"年份范围: {df['年份'].min()} - {df['年份'].max()}")
            logger.info(f"城市数量: {df['城市'].nunique()}")
            
            return df
            
        except Exception as e:
            logger.error(f"加载城市年度数据失败: {e}")
            raise
    
    def create_city_year_data_matrix(self, year: int = 2023) -> pd.DataFrame:
        """
        创建城市-年份数据矩阵
        
        Args:
            year: 目标年份
            
        Returns:
            城市数据矩阵
        """
        logger.info(f"正在创建{year}年城市数据矩阵")
        
        # 加载主数据
        df = self.load_main_data()
        
        # 筛选指定年份
        df_year = df[df['年份'] == year].copy()
        
        if df_year.empty:
            logger.warning(f"未找到{year}年的数据")
            return pd.DataFrame()
        
        # 设置索引
        df_year.set_index('城市', inplace=True)
        
        # 选择数值型列
        numeric_cols = df_year.select_dtypes(include=[np.number]).columns
        df_numeric = df_year[numeric_cols]
        
        logger.info(f"创建{year}年城市数据矩阵成功，形状: {df_numeric.shape}")
        
        return df_numeric
    
    def create_od_matrix_by_year(self, year: int = 2023, 
                                 value_col: str = '数据传输量_TB') -> pd.DataFrame:
        """
        创建指定年份的OD矩阵
        
        Args:
            year: 目标年份
            value_col: 用于创建矩阵的数值列
            
        Returns:
            OD矩阵DataFrame
        """
        logger.info(f"正在创建{year}年的OD矩阵，使用列: {value_col}")
        
        # 加载OD数据
        df = self.load_od_matrix()
        
        # 筛选指定年份
        df_year = df[df['年份'] == year].copy()
        
        if df_year.empty:
            logger.warning(f"未找到{year}年的OD数据")
            return pd.DataFrame()
        
        # 创建OD矩阵
        od_matrix = df_year.pivot_table(
            index='起点城市',
            columns='终点城市',
            values=value_col,
            aggfunc='sum'
        )
        
        # 确保所有城市都在矩阵中
        cities = self.config['data']['cities']
        od_matrix = od_matrix.reindex(index=cities, columns=cities, fill_value=0)
        
        logger.info(f"创建{year}年OD矩阵成功，形状: {od_matrix.shape}")
        
        return od_matrix
    
    def get_data_summary(self) -> Dict:
        """
        获取数据摘要
        
        Returns:
            数据摘要字典
        """
        logger.info("正在生成数据摘要")
        
        summary = {
            "project_root": self.project_root,
            "data_dir": self.data_dir
        }
        
        try:
            # 尝试加载数据配置
            data_config_path = os.path.join(self.project_root, 'config', 'data_config.json')
            if os.path.exists(data_config_path):
                with open(data_config_path, 'r', encoding='utf-8') as f:
                    summary['data_config'] = json.load(f)
        except Exception as e:
            logger.warning(f"无法加载数据配置: {e}")
        
        try:
            # OD矩阵数据摘要
            od_data = self.load_od_matrix()
            summary['od_matrix'] = {
                "total_records": len(od_data),
                "years": sorted(od_data['年份'].unique()),
                "cities": sorted(od_data['起点城市'].unique()),
                "columns": od_data.columns.tolist(),
                "missing_values": od_data.isnull().sum().to_dict()
            }
            
            # 城市年度数据摘要
            main_data = self.load_main_data()
            summary['main_data'] = {
                "total_records": len(main_data),
                "years": sorted(main_data['年份'].unique()),
                "cities": sorted(main_data['城市'].unique()),
                "columns": main_data.columns.tolist(),
                "missing_values": main_data.isnull().sum().to_dict()
            }
            
            logger.info("数据摘要生成成功")
            
        except Exception as e:
            logger.error(f"生成数据摘要失败: {e}")
            summary['error'] = str(e)
        
        return summary


# 使用示例
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建数据加载器
    loader = DataLoader(project_root=os.path.join(os.getcwd(), "粤港澳数据要素流动实验"))
    
    # 获取数据摘要
    summary = loader.get_data_summary()
    
    print("=" * 60)
    print("数据加载器测试")
    print("=" * 60)
    
    if 'od_matrix' in summary:
        print(f"OD矩阵数据:")
        print(f"  • 记录数: {summary['od_matrix']['total_records']}")
        print(f"  • 年份: {summary['od_matrix']['years']}")
        print(f"  • 城市数: {len(summary['od_matrix']['cities'])}")
    
    if 'main_data' in summary:
        print(f"\n城市年度数据:")
        print(f"  • 记录数: {summary['main_data']['total_records']}")
        print(f"  • 年份: {summary['main_data']['years']}")
        print(f"  • 城市数: {len(summary['main_data']['cities'])}")
        print(f"  • 指标数: {len(summary['main_data']['columns'])}")
    
    print("\n数据加载器测试完成!")
