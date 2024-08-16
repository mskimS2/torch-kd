# Torch-KD
This project offers a framework for applying `Knowledge Distillation (KD)` to various deep learning models. In KD, a large model (`Teacher`) helps a smaller model (`Student`) become more efficient while maintaining performance. This approach is ideal for creating models that work well in resource-constrained environments like mobile devices or embedded systems.

In this project, we demonstrate the application of Knowledge Distillation using CNN as the `Teacher` model and a simplified CNN model as the `Student` on the Image dataset (CIFAR-10). The framework is designed to be easily extensible to different datasets and models.

# Getting Started
You can easily manage dependencies using Poetry. Install all required packages by running:
```
poetry install
```
# Prepare the Dataset
This project uses the CIFAR-10 dataset, which is automatically downloaded via the torchvision library. No separate download process is required.

# Visualize Training Metrics with TensorBoard
```
poetry run tensorboard --logdir logs/
```

# Experimental Results: Performance Comparison
| Teacher (Size)        | Student (Size)              | KD (parameter value)            | Accuracy | Epoch |
|-----------------------|-----------------------------|---------------------------------|----------|-------|
| CNN (4.53M, baseline) | -                           | -                               | 82.80%   | 50    |
| -                     | SimpleCNN (1.02M, baseline) | -                               | 73.12%   | 50    |
| CNN (4.53M)           | SimpleCNN (1.02M)           | logits (weight=1.0)             | 73.59%   | 50    |
| CNN (4.53M)           | SimpleCNN (1.02M)           | logits (weight=0.1)             | 74.62%   | 50    |
| CNN (4.53M)           | SimpleCNN (1.02M)           | soft_target (T=1.0, weight=1.0) | 74.22%   | 50    |
| CNN (4.53M)           | SimpleCNN (1.02M)           | soft_target (T=4.0, weight=1.0) | 75.60%   | 50    |