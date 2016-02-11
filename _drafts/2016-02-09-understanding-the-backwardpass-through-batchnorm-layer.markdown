---
layout: "post"
title: "Understanding the backwardpass through Batchnorm-Layer"
date: "2016-02-09 14:54"
excerpt: "An explenation of gradient flow through Batchnorm-Layer following the circuit represantation learned in Standfords class CS231n "
---
At the moment there is a wonderful course running at Standford University, called [CS231n - Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/), held by Andrej Karpathy, Justin Johnson and Fei-Fei Li. Fortunately all the [course material](http://cs231n.stanford.edu/syllabus.html) is provided for free and all the lectures are recorded and uploaded on [Youtube](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC). This class gives and wonderful intro to machine learning/deep learning coming along with programming assignments.

## Batch Normalization

One Topic, which kept me quite buisy for some time was the implemantation of [Batch Normalization](http://arxiv.org/abs/1502.03167), especially the backward pass. Batch Normalization is a technique to provide any layer in a Neural Network inputs that are zero mean/unit variance - and this is basically what they like! But BatchNorm consists of one more step with makes this algorithm really powerful. Here is the the BatchNorm Algorithm:

<div class="fig figcenter fighighlight">
  <img src="/images/bn_backpass/bn_algorithm.PNG" width=400>
  <div class="figcaption"> Algorithm of Batch Normalization copied from the Paper by Ioffe and Szegedy mentioned above.<br><br>
  </div>
</div>


Look at the last line of the Algorithm. After normalizing the input `x` the result is squashed through a linear function with parameters `gamma` and `beta`. These are learnable parameters of the BatchNorm Layer and make it basically possible to say "Hey!! I dont want zero mean/unit variance input, give me back the raw input - it's better for me." If `gamma = sqrt(var(x))` and `beta = mean(x)`, the original activation is restored. This is, what makes BatchNorm really powerful in my opinion. We initialize the BatchNorm Parameters to transform the input to zero mean/unit variance distributions but during training they can learn that any other distribution might be better.
Anyway, I don't want to spend to much time on explaining Batch Normalization. If you want to learn more about it, the [paper](http://arxiv.org/abs/1502.03167) is very well written and [here](https://youtu.be/gYpoJMlgyXA?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&t=3078) Andrej is explaining BatchNorm in class.

Btw: it's called "Batch" Normalization because we perform this transformation and calculate the statistics only for a subpart (a batch) of the entire trainingsset.

## Backpropagation

In this blog post I don't want to give a lecture in Backpropagation and Stochastic Gradient Descent (SGD). For now I will assume that whoever will read this post, has some basic understanding of these principles. For the rest, let me quote Wiki:

>Backpropagation, an abbreviation for "backward propagation of errors", is a common method of training artificial neural networks used in conjunction with an optimization method such as gradient descent. The method calculates the gradient of a loss function with respect to all the weights in the network. The gradient is fed to the optimization method which in turn uses it to update the weights, in an attempt to minimize the loss function.

Uff, sounds tough, eh? I will maybe write another post about this topic but for now I want to focus on the concrete example of the backwardpass through the BatchNorm-Layer.

## Computational Graph of Batch Normalization Layer

I think one of the things I learned from the cs231n class that helped me most understanding backpropagation was the explenation through computational graphs. These Graphs are a good way to visualize the computational flow of fairly complex functions by small, piecewise differentiable subfunctions. For the Batch Normalization Layer it would look something like this:

<div class="fig figcenter fighighlight">
  <img src="/images/bn_backpass/BNcircuit.png">
  <div class="figcaption"> Computational graph of the BatchNorm-Layer. From left to right, following the black arrows flows the forward pass. The inputs are a matrix X and gamma and beta as vectors. From right to left, following the red arrows flows the backward pass which distributes the gradient from above layer to gamma and beta and all the way back to the input.<br><br>
  </div>
</div>

I think for all, who followed the course or who know the technique the forwardpass (black arrows) is easy and straightforward to read. From input `x` we calculate the mean of every dimension in the feature space and then substract this vector of mean values from every training example. With this done, following the lower branch, we calculate the per-dimension variance and with that the entire denominator of the normalization equation. Next we invert it and multiply it with differenz of inputs and means and we have `x_normalized`. The last two blobs on the left perform the squashing by multiplying with the input `gamma` and finally adding `beta`. Et voilà, we have our Batch-Normalized output.

A vanilla implemantation of the forwardpass might look like this:

```python
def batchnorm_forward(x, gamma, beta, eps):

  N, D = x.shape

  #step1: calciulate mean
  mu = 1./N * np.sum(x, axis = 0)

  #step2: substract mean vector of every trainings example
  xmu = x - mu

  #step3: following the lower branch - calculation denominator
  sq = xmu ** 2

  #step4: calculate variance
  var = 1./N * np.sum(sq, axis = 0)

  #step5: add eps for numerical stability, then sqrt
  sqrtvar = np.sqrt(var + eps)

  #step6: invert sqrtwar
  ivar = 1./sqrtvar

  #step7: execute normalization
  xhat = xmu * ivar

  #step8: Nor the two transformation steps
  gammax = gamma * xhat

  #step9
  out = gammax + beta

  #store intermediate
  cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)

  return out, cache
```

Note that for the excercise of the cs231n class we had to do a little more (calculate running mean and variance as well as implement different forward pass for trainings mode and test mode) but for the explenation of the backwardpass this piece of code will work.
In the cache variable we store some stuff that we need for the computing of the backwardpass, as you will see now!

## The power of Chain Rule for backpropagation

For all who kept on reading until now (congratualations!!), we close to arrive the backward pass of the BatchNorm-Layer.
To fully understand the channeling of the gradient backwards through the BatchNorm-Layer you should have some basic understanding of what the [Chain rule](https://en.wikipedia.org/wiki/Chain_rule) is. As a little refresh follows one figure that examplifies the use of chain rule for the backward pass in computational graphs.

<div class="fig figcenter fighighlight">
  <img src="/images/bn_backpass/chainrule_example.PNG">
  <div class="figcaption">The forwardpass on the left in calculates `z` as a function `f(x,y)` using the input variables `x` and `y` (This could literally be any function, examples are shown in the BatchNorm-Graph above). The right side of the figures shows the backwardpass. Recieving `dL/dz`, the gradient of the loss function with respect to `z` from above, the gradients of `x` and `y` on the loss function can be calculate by applying the chain rule, as shown in the figure.<br><br>
  </div>
</div>

So again, we only have to multiply the local gradient of the function with the gradient of above to channel the gradient backwards. Some derivations of some basic functions are listed in the [course material](http://cs231n.github.io/optimization-2/#sigmoid). If you understand that, and with some more basic knowledge in calculus, what will follow is a piece of cake!

# Finally: The Backpass of the Batch Normalization

In the comments of aboves code snippet I already numbered the computational steps by consecutive numbers. The Backpropagation follows these steps in reverse order. We will know take a more detailed look at every single computation of the backwardpass and by that deriving step by step a naive algorithm for the backward pass.

### Step 9

<div class="fig figleft fighighlight">
  <img src="/images/bn_backpass/step9.png">
  <div class="figcaption">Backwardpass through the last summation gate of the BatchNorm-Layer. Enclosured in brackets I put the dimensions of Input/Output<br><br>
  </div>
</div>
Recall that the derivation of a function `f = x + y` with respect to any of these two variables is `1`. This means to channel a gradient through a summation gate, we only need to multiply by `1`. And because the summation of `beta` during the forward pass is a row-wise summation, during the backward pass we need to sum up the gradient over all of its columns (take a look at the dimensions). So after the first step of backpropagation we already got the gradient for one learnable parameter: `beta`

### Step 8

<div class="fig figleft fighighlight">
  <img src="/images/bn_backpass/step8.png">
  <div class="figcaption">Next follows the backward pass through the multiplication gate of the normalized input and the vector of gamma.<br><br>
  </div>
</div>
For any function `f = x * y` the derivation with respect to one of the inputs is simply just the other input variable. This also means, that for this step of the backward pass we need the variables used in the forward pass of this gate (luckily stored in the `cache` of aboves function). So again we get the gradients of the two inputs of these gates by applying chain rule (  = multiplying the local gradient with the gradient from above). For `gamma`, as for `beta` in step 9, we need to sum up the gradients over dimension `N`, because the multiplication was again row-wise. So we now have the gradient for the second learnable parameter of the BatchNorm-Layer `gamma` and "only" need to backprop the gradient to the input `x`, so that we then can backpropagate the gradient to any layer further downwards.

### Step 7

<div class="fig figleft fighighlight">
  <img src="/images/bn_backpass/step7.png">
  <div class="figcaption">This step during the forward pass was the final step of the normalization combining the two branches (nominator and denominator) of the computational graph. During the backward pass we will calculate the gradients that will flow seperatly through these two branches backwards.<br><br>
  </div>
</div>
It's basically the exact same operation, so lets not waste much time and continue. The two needed variables `xmu` and `ivar` for this step are also stored `cache` variable we pass to the backprop function. (And again: This is one of the main advantages of computational graphs. Splitting complex functions into a handful of simple basic operations. And like this you have a lot of repetitions!)

### Step 6

<div class="fig figleft fighighlight">
  <img src="/images/bn_backpass/step6.png">
  <div class="figcaption">This is a "one input-one output" node where, during the forward pass, we inverted the input (square root of the variance)<br><br>
  </div>
</div>
The local gradient is visualized in the image and should not be hard to derive by hand. Multiplied by the gradient from above is what we channel to the next step. `sqrtvar` is also one of the variables passed in `cache`.

### Step 5

<div class="fig figleft fighighlight">
  <img src="/images/bn_backpass/step5.png">
  <div class="figcaption">Again "one input-one output". This node calculates during the forward pass the denominator of the normalization.<br><br>
  </div>
</div>
The derivation of the local gradient is little magic and should need no explenation. `var` and `eps` are also passed in the `cache`. No more words to lose!

### Step 4

<div class="fig figleft fighighlight">
  <img src="/images/bn_backpass/step4.png">
  <div class="figcaption">Also a "one input-one output" node. During the forward pass the output of this node is the variance of each feature `d for d in [1...D]`.<br><br>
  </div>
</div>
The derivation of this steps local gradient might look unclear at the very first glance. But it's not that hard at the end. Let's recall that a normal summation gate (see step 9) during the backward pass only transfers the gradient unchanged and evenly to the inputs. With that in mind, it should not be that hard to conclude, that a column-wise summation during the forward pass, during the backward pass means that we evenly distribute the gradient over all rows for each column. And not much more is done here. We create a matrix of ones with the same shape as the input `sq` of the forward pass, devide it element-wise by the number of rows (thats the local gradient) and multiply it by the gradient from above.

### Step 3

<div class="fig figleft fighighlight">
  <img src="/images/bn_backpass/step3.png">
  <div class="figcaption">This node outputs the square of its input, which during the forward pass was a matrix containing the input `x` subtracted by the per-feature `mean`.<br><br>
  </div>
</div>
I think for all who followed until here, there is not much to explain for the derivation of the local gradient.

### Step 2

<div class="fig figleft fighighlight">
  <img src="/images/bn_backpass/step2.png">
  <div class="figcaption">Now this looks like a more fun gate! two inputs-two outputs! This node substracts the per-feature mean row-wise of each trainings example `n for n in [1...N]` during the forward pass.<br><br>
  </div>
</div>
Okay lets see. One of the definitions of backprogatation and computational graphs is, that whenever we have two gradients coming to one node, we simply add them up. Knowing this, the rest is little magic as the local gradient for a substraction is as hard to derive as for a summation. Note that for `mu` we have to sum up the gradients over the dimension `N` (as we did before for `gamma` and `beta`).

### Step 1
<div class="fig figleft fighighlight">
  <img src="/images/bn_backpass/step1.png">
  <div class="figcaption">The function of this node is exactly the same as of step 4. Only that during the forward pass the input was `x` - the input to the BatchNorm-Layer and the output here is `mu`, a vector that contains the mean of each feature.<br><br>
  </div>
</div>
As this node executes the exact same operation as the one explained in step 4, also the backpropagation of the gradient looks the same. So let's continue to the last step.

### Step 0 - Arriving at the Input

<div class="fig figcenter fighighlight">
  <img src="/images/bn_backpass/step0.png">
  <div class="figcaption"><br><br>
  </div>
</div>
I only added this image to again visualize that at the very end we need to sum up the gradients `dx1` and `dx2` to get the final gradient `dx`. This matrix contains the gradient of the loss function with respect to the input of the BatchNorm-Layer. This gradient `dx` is also what we give as input to the backwardpass of the next layer, as for this layer we recieve `dout` from the layer above.
