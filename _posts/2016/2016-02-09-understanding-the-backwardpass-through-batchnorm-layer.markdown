---
layout: "post"
title: "Understanding the backwardpass through Batchnorm-Layer"
date: "2016-02-09 14:54"
excerpt: "An explenation of gradient flow through Batchnorm-Layer following the circuit represantation learned in Standfords class CS231n "
---
At the moment there is a wonderful course running at Standford University, called [Cs231n - Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/), held by Andrej Karpathy, Justin Johnson and Fei-Fei Li. Fortunately all the [course material](http://cs231n.stanford.edu/syllabus.html) is provided for free and all the lectures are recorded and uploaded on [Youtube](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC). This class gives and wonderful intro to machine learning/deep learning coming along with programming assignments.

## Batch Normalization

One Topic, which kept me quite buisy for some time was the implemantation of [Batch Normalization](http://arxiv.org/abs/1502.03167), especially the backward pass. Batch Normalization is a technique to provide any layer in a Neural Network inputs that are zero mean/unit variance - and this is basically what they like! But BatchNorm consists of one more step with makes this algorithm really powerful. Here is the the BatchNorm Algorithm:

<div class="fig figcenter fighighlight">
  <img src="/images/bn_backpass/bn_algorithm.PNG" width=400>
  <div class="figcaption"> Algorithm of Batch Normalization copied from the Paper by Ioffe and Szegedy mentioned above.<br><br>
  </div>
</div>


Look at the last line of the Algorithm. After normalizing the input `x` the result is squashed through a linear function with parameters `gamma` and `beta`. These are learnable parameters of the BatchNorm Layer and make it basically possible to say "Hey!! I dont want zero mean/unit variance input, give me back the raw input - it's better for me." If `gamma = sqrt(var(x))` and `beta = mean(x)`, the original activation is restored. This is, what makes BatchNorm really powerful in my opinion. We initialize the BatchNorm Parameters to transform the input to zero mean/unit variance distributions but during training they can learn that any other distribution might be better.
Anyway, I don't want to spend to much time on explaining Batch Normalization. If you want to learn more about it, the [paper](http://arxiv.org/abs/1502.03167) is very well written and [here](https://youtu.be/gYpoJMlgyXA?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&t=3078) Andrej is explaining BatchNorm in class.

## Backpropagation

In this blog post I don't want to give a lecture in Backpropagation. But since this post is about Backpropagation in the BatchNorm Layer i will briefly try to explain what Backpropagation is or for what it is used. So let's first see what Wiki says:

>Backpropagation, an abbreviation for "backward propagation of errors", is a common method of training artificial neural networks used in conjunction with an optimization method such as gradient descent. The method calculates the gradient of a loss function with respect to all the weights in the network. The gradient is fed to the optimization method which in turn uses it to update the weights, in an attempt to minimize the loss function.

Uff, sounds tough. So what do we want? We have a model (e.g. Neural Network) with some model parameters (e.g. the weights of any Layer or `gamma` and `beta` for the case of BatchNorm) in which we feed some input (e.g. an image) and receive and output (e.g. class of object in that image). During training we feed data through the Network and quantify the output by any loss function. In general they are defined in the way that we want the loss function to be minimized. So while training we try to find a set of model parameters that minimizes the loss function, or better, that gives us the best result for our task. And luckily we don't need to search blindly in the endless space of parameter combinations, instead we just calculate the gradient of the loss function and change our model parameters by some amount in the opposite direction. Wait...what?
Most of the time, this is explained in 2D-Space. Assume that our model has 2 parameters. The loss function as a function of the model parameters can be plotted as a 3D-surface plot.

<div class="fig figcenter fighighlight">
  <img src="/images/bn_backpass/peaks.png">
  <div class="figcaption"> Algorithm of Batch Normalization copied from the Paper by Ioffe and Szegedy mentioned above.<br><br>
  </div>
</div
