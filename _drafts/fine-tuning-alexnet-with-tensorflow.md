---
layout: "post"
title: "Fine-tuning AlexNet with Tensorflow"
date: "2017-02-21 18:00"
excerpt: "This blog post will guide you on how to finetune AlexNet with pure Tensorflow."
---

After over one year I finally found time and leisure to write my next article. This time, I will write about finetuning AlexNet in pure [Tensorflow 1.0](https://www.tensorflow.org/install/migration).
You might wonder, why there is a need for another article covering finetuning convolutional neural networks. Actually a really good question, and here is my answer:

- Albeit there exist many How-To's, most of the newer once are covering finetuning VGG or Inception Models and not AlexNet. Although the idea behind finetuning is the same, the major difference is, that Tensorflow (as well as Keras) already ship with VGG or Inception classes and include the weights (pretrained on ImageNet). For the AlexNet model, we have to do a bit more on our own.
- Another reason is that for a lot of my personal projects AlexNet works quite well and there is no reason to switch to any of the more heavy-weight models to gain probably another .5% accuracy boost. As the models get deeper they naturally need more computational time, which in some projects I can't afford.
- In the past, I used mainly [Caffe](http://caffe.berkeleyvision.org/) to finetune convolutional networks. But to be honest, I found it quite cumbersome (e.g. model definition via .prototxt, the model structure with blobs...) to work with Caffe. After some time with Keras, I recently switched to pure Tensorflow and now I want to be able to finetune the same network as previously, but using just Tensorflow.
- Well and I think the main reason for this article is that working on a project like this, helps me to better understand Tensorflow in general.

## Preliminary Information
This will not be a beginner guide neither on finetuning nor on Tensorflow and I will assume that whoever reads this, has basic knowledge on deep learning, Python and Tensorflow.

## Getting the weights and model structure
