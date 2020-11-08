# Neural Networks

## Neural networks: without the brain stuff

"Neural Network" is a very board term; these are more accurately called
"fully-connected networks" or sometimes "multi-layer perceptrons"(MLP)

## Activation functions

ReLU is a good default choice for most problems

## Neural Networks: Architectures

网络的称呼，忽略 input layer。
可以称 2 层神经网络，或一个隐藏层神经网络。

sigmod 函数的导数
$$ (\frac{1}{1+e^{(-x)}})' = \frac{1}{1+e^{(-x)}} * (1-\frac{1}{1+e^{(-x)}}) $$

P27 ???
***Todo***

Do not use size of neural network as a regularizer. Use stronger regularization instead.

## Problem: How to compute gradients

P38
***Todo***
loss 采用 svm+regularization

Upstream gradient:
$\frac{\partial L}{\partial z}$
local gradient:
$\frac{\partial z}{\partial x}$, $\frac{\partial z}{\partial y}$
Downstream gradient:
$\frac{\partial L}{\partial x}$

## Patterns in gradient flow

## Backprop Implementation: Modularized API

Backprop Implementation: Modularized API

```python
class ComputationalGraph(object):
    #...
    def forward(inputs):
        # 1. [pass inputs to input gates...]
        # 2. forward the computational graph:  
        for gate in self.graph.nodes_topologically_sorted():
            gate.forward()
        return loss # the final gate in the graph outputs the loss
    def backward():
        for gate in reversed(self.graph.nodes_topologically_sorted()):
            gate.backward() # little piece of backprop (chain rule applied)
        return inputs_gradients
```

Forward and Backprop implementation in pytorch  

Backprop with Matrices(or Tensors)


