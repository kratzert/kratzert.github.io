---
layout: "post"
title: "Understanding the gradient flow through the Batch Normalization Layer"
date: "2016-02-12 13:56"
excerpt: "Describing the naive backward pass of the gradient through the BatchNorm-Layer by looking at the computational graph."
comments: true
---

At the moment there is a wonderful course running at Standford University, called [CS231n - Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/), held by Andrej Karpathy, Justin Johnson and Fei-Fei Li. Fortunately all the [course material](http://cs231n.stanford.edu/syllabus.html) is provided for free and all the lectures are recorded and uploaded on [Youtube](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC). This class gives and wonderful intro to machine learning/deep learning coming along with programming assignments.

## Batch Normalization

One Topic, which kept me quite buisy for some time was the implemantation of [Batch Normalization](http://arxiv.org/abs/1502.03167), especially the backward pass. Batch Normalization is a technique to provide any layer in a Neural Network with inputs that are zero mean/unit variance - and this is basically what they like! But BatchNorm consists of one more step with makes this algorithm really powerful. Here is the the BatchNorm Algorithm:

<div class="fig figcenter fighighlight">
  <img src="/images/bn_backpass/bn_algorithm.PNG">
  <div class="figcaption"><br> Algorithm of Batch Normalization copied from the Paper by Ioffe and Szegedy mentioned above.<br>
  </div>
</div>


Look at the last line of the Algorithm. After normalizing the input `x` the result is squashed through a linear function with parameters `gamma` and `beta`. These are learnable parameters of the BatchNorm Layer and make it basically possible to say "Hey!! I dont want zero mean/unit variance input, give me back the raw input - it's better for me." If `gamma = sqrt(var(x))` and `beta = mean(x)`, the original activation is restored. This is, what makes BatchNorm really powerful in my opinion. We initialize the BatchNorm Parameters to transform the input to zero mean/unit variance distributions but during training they can learn that any other distribution might be better.
Anyway, I don't want to spend to much time on explaining Batch Normalization. If you want to learn more about it, the [paper](http://arxiv.org/abs/1502.03167) is very well written and [here](https://youtu.be/gYpoJMlgyXA?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&t=3078) Andrej is explaining BatchNorm in class.
