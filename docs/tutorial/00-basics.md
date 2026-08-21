# Minimum Concepts

[Next: environment](01-environment.md) | [Tutorial home](README.md)

A classifier maps an image tensor to one logit for each class. Softmax turns
logits into probabilities; cross entropy compares those probabilities with the
integer label. Backpropagation computes parameter gradients and the optimizer
updates the model.

You do not need to implement a CNN before completing this project. Read
[examples/01_tensor_flow.py](../../examples/01_tensor_flow.py) and
[examples/03_minimal_training.py](../../examples/03_minimal_training.py) to
see the smallest runnable form of these steps.

The project distinguishes three data splits: train updates parameters, valid
selects a checkpoint and test is held out for the final report. Do not tune
settings against test metrics.
