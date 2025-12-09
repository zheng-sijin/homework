# 创建README.md
readme_content = """# 粤港澳大湾区数据要素流动多元统计分析实验

## 项目简介
本项目基于粤港澳大湾区2019-2023年数据要素流动数据，运用多元统计分析方法进行深入研究，包括：
- 主成分分析(PCA)与因子分析
- 聚类分析(K-Means, 层次聚类等)
- 空间计量分析
- 网络分析
- 机器学习预测

## 数据说明
### 主要数据集
1. **OD矩阵数据** (`od_matrix.csv`): 包含2019-2023年11个城市间的双向数据流动
   - 数据传输量(TB)
   - API调用频次(万次)
   - 企业合作数据项目数

2. **城市年度数据** (`main_data_advanced.csv`): 包含各城市年度指标
   - 数据流动维度(18个指标)
   - 经济发展维度(12个指标)
   - 创新能力维度(10个指标)
   - 基础设施维度(8个指标)

## 项目结构
粤港澳数据要素流动实验/
├── README.md
├── requirements.txt
├── environment.yml
├── config/
├── data/
├── src/
├── notebooks/
├── scripts/
├── tests/
├── outputs/
└── docs/

text

## 快速开始
### 1. 环境配置
```bash
# 使用conda创建环境
conda env create -f environment.yml

# 激活环境
conda activate gba-data-analysis

# 或使用pip安装依赖
pip install -r requirements.txt