# 滚动轴承故障诊断研究

基于多尺度特征融合与注意力机制的滚动轴承故障诊断方法研究

## 项目结构

```
├── data/
│   ├── raw/              # 存放下载的.mat文件
│   └── processed/        # 存放预处理后的.npy文件
├── models/               # 保存模型权重
├── results/              # 保存实验图表
├── src/
│   ├── preprocess.py     # 数据预处理
│   ├── dataset.py        # 数据集类
│   ├── model.py          # 模型定义（基线+4个改进）
│   ├── train.py          # 训练脚本
│   ├── evaluate.py       # 评估脚本
│   └── utils.py          # 工具函数
└── main.py               # 主入口
```

## 模型进展

- **Model-0**: 基线1D-CNN - 基准性能
- **Model-1**: + 多尺度卷积 - 改进一
- **Model-2**: + 通道注意力 - 改进二
- **Model-3**: + 空间注意力 - 改进三
- **Model-4**: + 多层级特征融合 - 改进四
- **Model-Full**: 全部四个改进
