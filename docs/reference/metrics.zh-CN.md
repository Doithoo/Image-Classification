# 指标与评测证据

[English](metrics.md) | [评测教程](../tutorial/05-evaluation-and-deployment.zh-CN.md)

混淆矩阵的行是真实类别、列是预测类别。accuracy 按样本频率加权常见类；balanced accuracy 只对当前 split 中实际出现的类别平均 recall。macro F1 对出现的类别等权；weighted F1 按数据分布加权。

有概率时，评测还记录 top-5 accuracy、NLL、Brier score、expected calibration error 和 calibration bins。输出目录包含：

```text
evaluation.json  schema、数据/checkpoint identity 和聚合指标
predictions.csv  全部预测、置信度、真实类概率、熵和 top-k
errors.csv       按预测置信度排序的误分类
per_class.csv    precision、recall、F1 和 support
calibration.csv  可靠性分箱证据
```

模型选择使用验证集证据。测试集证据只用于一次最终固定比较，不应用于反复调参。
