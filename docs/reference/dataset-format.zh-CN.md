# 数据格式与身份

[English](dataset-format.md) | [数据教程](../tutorial/02-data.zh-CN.md)

`prepare-data` 会原子发布以下文件：

```text
dataset.yaml   schema-v2 数据集契约和组合 identity
train.csv      path,label 行
valid.csv
test.csv
source.yaml    兼容旧读取方的子集
summary.txt    人可读摘要
```

CSV 路径是相对于 `data_root` 的 POSIX 路径；label 是指向有序 `classes` 的连续零基索引。`dataset.yaml` 记录每个 split 的完整 SHA-256、每类数量、源图像聚合 SHA-256、切分策略和由它们共同派生的 `identity`。

`verify-data` 会校验所有文件、label、源图像字节和组合 identity。数据改变未必是错误，但它会形成新的数据契约；在声明指标前应审阅并重新准备 manifest。
