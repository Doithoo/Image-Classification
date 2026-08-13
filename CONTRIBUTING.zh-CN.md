# 贡献指南

[English](CONTRIBUTING.md)

## 环境

```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
pre-commit install
```

请在独立分支提交范围明确的修改。行为变化必须配测试；文档命令应能从仓库根目录
执行。不要提交数据集、训练产物、凭据或生成的模型文件。

## 验证

```bash
ruff check .
ruff format --check .
pytest
```

目录职责和测试与源码的对应关系见 [`scripts/README.zh-CN.md`](scripts/README.zh-CN.md)、
[`src/README.zh-CN.md`](src/README.zh-CN.md) 和 [`tests/README.zh-CN.md`](tests/README.zh-CN.md)。

数据或实验修改需记录源压缩包校验和、数据质量补丁版本、manifest 校验和、完整配置、
依赖锁与精确命令。只用验证集选择模型，不要用测试集调参。

提交信息使用英文 ASCII 与
[Conventional Commits](https://www.conventionalcommits.org/) 格式，例如
`fix(data): Reject conflicting duplicate labels`。主题不超过 72 字符，说明使用首字母
大写的祈使句，不以句号结尾。

行为错误请使用 bug 表单，学习材料不清楚请使用 learning 表单，代码修改请使用 PR
模板。
