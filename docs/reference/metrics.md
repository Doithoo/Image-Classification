# Metrics And Evaluation Evidence

[简体中文](metrics.zh-CN.md) | [Evaluation tutorial](../tutorial/05-evaluation-and-deployment.md)

Evaluation uses rows=true and columns=predicted confusion matrices. Accuracy
weights common classes by frequency; balanced accuracy averages recall only
across classes represented in the evaluated split. Macro F1 weights represented
classes equally; weighted F1 follows the data distribution.

When probabilities are available, evaluation also records top-5 accuracy, NLL,
Brier score, expected calibration error and calibration bins. The output
directory contains:

```text
evaluation.json  schema, dataset/checkpoint identities and aggregate metrics
predictions.csv  all predictions, confidence, true probability, entropy, top-k
errors.csv       misclassifications sorted by predicted confidence
per_class.csv    precision, recall, F1 and support
calibration.csv  reliability-bin evidence
```

Use validation evidence for model selection. Test evidence belongs to one final
fixed comparison, not an iterative tuning loop.
