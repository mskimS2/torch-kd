# Torch-KD
This project offers a framework for applying `Knowledge Distillation(KD)` to various deep learning models. In KD, a large model (`Teacher`) helps a smaller model (`Student`) become more efficient while maintaining performance. This approach is ideal for creating models that work well in resource-constrained environments like mobile devices or embedded systems.

In this project, we demonstrate the application of Knowledge Distillation using CNN as the `Teacher` model and a simplified `CNN`, `ResNet` model as the `Student` on the Image dataset (CIFAR-10). The framework is designed to be easily extensible to different datasets and models.

# Getting Started
You can easily manage dependencies using Poetry. Install all required packages by running:
```
poetry shell
poetry install
```
# Prepare the Dataset
This project uses the CIFAR-10 dataset, which is automatically downloaded via the torchvision library. No separate download process is required.

# Visualize Training Metrics with TensorBoard
```
tensorboard --logdir=./logs
```

# Experimental Results: Performance Comparison
The Teacher model and Student model are simple model structures inspired by [`CNN`](https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html) and [`ResNet`](https://arxiv.org/pdf/1512.03385), respectively, with the model size differing by approximately 4 times.

| Teacher (Size)              | Student (Size)              | KD (parameter value)            | Accuracy | Epoch |
|-----------------------------|-----------------------------|---------------------------------|----------|-------|
| ResNet (4.57MB, baseline)   | -                           | -                               | 92.84%   | 200   |
| -                           | ResNet (1.19MB, baseline)   | -                               | 83.61%   | 30    |
| ResNet (4.57MB, predtrained)| ResNet (1.19MB)             | logits (weight=1.0)             | 87.51%   | 30    |
| ResNet (4.57MB, predtrained)| ResNet (1.19MB)             | soft_target (T=4.0, weight=1.0) | 86.69%   | 30    |
| ResNet (4.57MB, predtrained)| ResNet (1.19MB)             | hints (weight=1.0)              | 87.90%   | 30    |
| ResNet (4.57MB, predtrained)| ResNet (1.19MB)             | attention_transfer (weight=1.0) | 86.21%   | 30    |