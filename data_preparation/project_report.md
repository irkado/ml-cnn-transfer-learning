---
header-includes:
  - '\usepackage{ragged2e}'
  - '\justifying'
---
# Image Classification Using Transfer Learning

## 1. Introduction

==Image classification is a core problem in computer vision, where the goal is to assign a semantic label to an input image. Modern solutions rely on Convolutional Neural Networks, which are able to learn hierarchical feature representations directly from raw pixel data. While deep CNNs achieve strong performance, training them from scratch typically requires large labeled datasets and substantial computational resources.==

==Transfer learning mitigates these limitations by reusing models pretrained on large-scale datasets such as ImageNet. Instead of learning all features from scratch, pretrained networks provide generic low- and mid-level visual representations that can be adapted to new tasks with relatively little data and training time.==

In this project, transfer learning is applied to image classification on the CIFAR-10 and CIFAR-100 datasets. Two architectures with different design goals are evaluated: ResNet-18, a residual network emphasizing representational depth, and MobileNetV2, a lightweight architecture optimized for efficiency. Multiple fine-tuning strategies are compared to analyze their impact on convergence speed, generalization, and robustness across datasets of differing complexity.

---
## 2. Problem Definition

The task addressed in this project is supervised multi-class image classification. Given an input image, the model must predict exactly one class label from a predefined set of categories.

We focused on the following questions:
- How does the depth and capacity of the backbone architecture affect transfer learning performance?
- How do different fine-tuning strategies (head-only, partial fine-tuning, full fine-tuning) influence accuracy and generalization?
- How does dataset complexity, as represented by CIFAR-10 versus CIFAR-100, impact transfer learning outcomes?

To answer them, all models are trained using identical data preprocessing pipelines and evaluated using consistent quantitative and qualitative metrics.

---
## 3. Datasets

Experiments are conducted on the CIFAR-10 and CIFAR-100 datasets. Both datasets consist of RGB images with an original resolution of 32×32 pixels.

CIFAR-10 contains 10 classes with 5,000 images per class, while CIFAR-100 contains 100 classes with 500 images per class. Although both datasets have similar total sizes, CIFAR-100 is significantly more challenging due to the increased number of classes and higher inter-class similarity.

To enable transfer learning from ImageNet-pretrained models, all images are resized to 224×224 pixels during data preprocessing. Training data undergoes data augmentation consisting of random horizontal flips and random cropping with padding. All images are normalized using ImageNet mean and standard deviation values. Here we can observe tha main difference between picture from dataset and the image to which transformation was applied : 

![alt text](cifar_10_visual_difference.png)

By means of blurring we reduce the sharp lines and corners on the original picture, consequently our model will learn visual pattern way more better.

There are also few examples of our images from CIFAR-10 after augumentation phase :

![alt text](image_1.png)

For CIFAR-100 :

![alt text](cifar_100_image_1.png)

Moreover, after perfoming Normalization the intensity of our RGB channels is approximately the same, which means that our pictures have no preferences in red, green or blue colors :

For CIFAR-10 :

![alt text](image_3.png) 

For CIFAR-100 :

![alt text](pixel_distrubution_cifar_100-1.png)

The training set is split into training and validation subsets using a fixed random seed to ensure reproducibility. The final evaluation is performed on a held-out test set that is never seen during training or model selection.

The quantity of instances in traing/validation/test sets for CIFAR-10 :

![alt text](cifar_10_train_validation_test_lables.png)

For CIFAR-100 in training loader : 

![alt text](labels_count_cifar_100.png)

As it can be seen from table above, we have approximately the same amount of instances for each class, consequently our model will generalize well and focus not on specific class, but rather on the whole classes.

---
## 4. Model Architecture

Two convolutional neural network architectures are evaluated: ResNet-18 and MobileNetV2. Both models are initialized with pretrained ImageNet weights.

ResNet-18 is a residual network that uses skip connections to facilitate gradient flow and enable effective training of deeper architectures. The final fully connected layer is replaced with a task-specific classification head consisting of a dropout layer followed by a linear layer.

MobileNetV2 is a lightweight architecture designed for efficiency. It employs depthwise separable convolutions and inverted residual blocks to reduce computational cost. Similar to ResNet-18, its original classifier is replaced with a dropout layer followed by a linear classification layer.

For both architectures, dropout is applied in the classification head, with the dropout probability gradually reduced as more layers are unfrozen during fine-tuning. This design balances regularization during early training with increased representational capacity during later fine-tuning stages.

---
## 5. Training

Training is performed using supervised learning with categorical cross-entropy loss. All experiments use the Adam optimizer, with learning rates carefully adjusted depending on the fine-tuning strategy to avoid catastrophic forgetting of pretrained weights.

Three transfer learning strategies are evaluated:

1. **Head-only training**  
   The backbone network is fully frozen, and only the classification head is trained. A learning rate of 1e-3 is used to allow rapid adaptation to the target dataset. Training is performed for 5 epochs.

2. **Last-block fine-tuning**  
   In addition to the classification head, the final convolutional block of the backbone is unfrozen. This allows higher-level features to adapt to the target task while preserving lower-level representations. A reduced learning rate of 1e-4 is used, and training is again performed for 5 epochs.

3. **Full model fine-tuning**  
   All layers of the network are unfrozen, enabling full adaptation of the pretrained model. To prevent destabilizing previously learned features, a low learning rate of 1e-5 is used. Full fine-tuning is performed for 15 epochs.

Model performance is monitored using validation loss and validation accuracy. The model state with the highest validation accuracy is saved during each training phase. Final evaluation is performed on the test set using accuracy, precision, recall, and F1-score, along with qualitative analysis of prediction behavior and class confusion patterns.

## 6. Results
### 6.1 CIFAR-100 Model Comparison: ResNet-18 vs MobileNetV2

#### Accuracy Comparison

| ResNet-18                                                                                            | MobileNetV2                                                                                            |
| ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| ![](figures/cifar_100_resnet18/accuracy_bar_plot.png)                                                | ![](figures/cifar_100_mobilenet_v2/accuracy_bar_plot.png)                                              |
| **Figure 1(a):** Validation accuracy across transfer learning strategies for ResNet-18 on CIFAR-100. | **Figure 1(b):** Validation accuracy across transfer learning strategies for MobileNetV2 on CIFAR-100. |

As shown in **Figure 1(a)** and **Figure 1(b)**, ResNet-18 consistently achieves higher validation accuracy than MobileNetV2 across all fine-tuning strategies. The performance gap is most pronounced under full model fine-tuning, indicating that the deeper architecture of ResNet-18 benefits more from end-to-end adaptation. MobileNetV2, while exhibiting a lower accuracy ceiling, still demonstrates clear gains from fine-tuning, confirming the effectiveness of transfer learning for lightweight architectures.

---
#### Training Dynamics: Learning Speed

| ResNet-18                                                                             | MobileNetV2                                                                             |
| ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| ![](figures/cifar_100_resnet18/learning_speed.png)                                    | ![](figures/cifar_100_mobilenet_v2/accuracy_learning_speed.png)                         |
| **Figure 2(a):** Validation accuracy over training epochs for ResNet-18 on CIFAR-100. | **Figure 2(b):** Validation accuracy over training epochs for MobileNetV2 on CIFAR-100. |

In **Figure 2(b)** we can see that MobileNetV2 converges faster during the early training epochs, reflecting its reduced model complexity. In contrast, **Figure 2(a)** shows that ResNet-18 continues to improve over a longer training horizon and ultimately reaches a higher accuracy plateau. This highlights the trade-off between faster convergence and higher representational capacity.

---
#### Overfitting and Generalization

| ResNet-18                                                                                        | MobileNetV2                                                                                        |
| ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| ![](figures/cifar_100_resnet18/loss_curves.png)                                                  | ![](figures/cifar_100_mobilenet_v2/head_full_last_loss_curves.png)                                 |
| **Figure 3(a):** Training and validation loss and generalization gap for ResNet-18 on CIFAR-100. | **Figure 3(b):** Training and validation loss and generalization gap for MobileNetV2 on CIFAR-100. |

Compared to **Figure 3(b)**, **Figure 3(a)** shows that ResNet-18 achieves lower training loss but exhibits a larger generalization gap under full fine-tuning, indicating increased overfitting risk. MobileNetV2 displays more stable generalization behavior, likely due to its reduced parameter count.

---
#### Qualitative Evaluation

| ResNet-18                                                                              | MobileNetV2                                                                              |
| -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| ![](figures/cifar_100_resnet18/resnet_correct_wrong_labeled_images.png)                | ![](figures/cifar_100_mobilenet_v2/model_pred.png)                                       |
| **Figure 4(a):** Correct and incorrect predictions produced by ResNet-18 on CIFAR-100. | **Figure 4(b):** Correct and incorrect predictions produced by MobileNetV2 on CIFAR-100. |

Qualitative results in **Figure 4(a)** and **Figure 4(b)** indicate that both models correctly classify samples with clear visual structure. However, misclassifications occur more frequently for MobileNetV2, particularly for visually similar or low-resolution samples, reflecting its limited representational capacity.

---
#### Error Analysis: Class Confusion

##### Apple

| ResNet-18                                                                               | MobileNetV2                                                                               |
| --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| ![](figures/cifar_100_resnet18/predictions.png)                                         | ![](figures/cifar_100_mobilenet_v2/wrong_apple_predicted_analytics.png)                   |
| **Figure 5(a):** Predicted class distribution for true *apple* samples using ResNet-18. | **Figure 5(b):** Predicted class distribution for true *apple* samples using MobileNetV2. |

In **Figure 5(a)** and **Figure 5(b)** we can see that for both models misclassifications are largely restricted to semantically related food categories, indicating that learned representations capture meaningful high-level semantic structure.

---
##### Woman

| Misclassified Samples                                                              | Confusion Matrix                                                                  |
| ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| ![](figures/cifar_100_mobilenet_v2/images_of_wrong_labeled_women.png)              | ![](figures/cifar_100_mobilenet_v2/conf_matr_wrong_labeled_women.png)             |
| **Figure 6(a):** Examples of misclassified images where the true class is *woman*. | **Figure 6(b):** Confusion matrix for predictions when the true label is *woman*. |

As illustrated in **Figure 6(a)** and **Figure 6(b)**, confusion occurs primarily among closely related human categories such as *girl*, *man*, and *boy*, highlighting the difficulty of fine-grained human classification at low image resolution.

### 6.2 CIFAR-10 Model Comparison: ResNet-18 vs MobileNetV2

#### Accuracy Comparison

| ResNet-18                                                                                     | MobileNetV2                                                                                     |
| --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| ![](figures/cifar_10_resnet18/bar_plot_accuracy.png)                                          | ![](figures/cifar_10_mobilenet_v2/new_accuracy_bar_plot.png)                                    |
| **Figure 7(a):** Validation accuracy across fine-tuning strategies for ResNet-18 on CIFAR-10. | **Figure 7(b):** Validation accuracy across fine-tuning strategies for MobileNetV2 on CIFAR-10. |

As shown in **Figures 7(a)** and **7(b)**, both architectures achieve substantially higher validation accuracy on CIFAR-10 compared to CIFAR-100. Full model fine-tuning yields the best performance for both networks, with ResNet-18 achieving the highest overall accuracy. The reduced number of classes and higher inter-class separability in CIFAR-10 allow both models to generalize more effectively.

---
#### Training Dynamics: Learning Speed

| ResNet-18                                                                   | MobileNetV2                                                                   |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| ![](figures/cifar_10_resnet18/accuracy_learning_speed.png)                  | ![](figures/cifar_10_mobilenet_v2/new_accuracy_learning_speed.png)            |
| **Figure 8(a):** Validation accuracy over epochs for ResNet-18 on CIFAR-10. | **Figure 8(b):** Validation accuracy over epochs for MobileNetV2 on CIFAR-10. |

From **Figures 8(a)** and **8(b)**, both models exhibit rapid convergence on CIFAR-10. MobileNetV2 converges slightly faster during early epochs, while ResNet-18 continues improving over a longer training horizon and reaches a marginally higher accuracy ceiling. Compared to CIFAR-100, convergence is faster and more stable across all fine-tuning strategies.

---
#### Overfitting and Generalization

| ResNet-18                                                                                   | MobileNetV2                                                                                   |
| ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| ![](figures/cifar_10_resnet18/loss_curves.png)                                              | ![](figures/cifar_10_mobilenet_v2/new_loss_curves.png)                                        |
| **Figure 9(a):** Training/validation loss and generalization gap for ResNet-18 on CIFAR-10. | **Figure 9(b):** Training/validation loss and generalization gap for MobileNetV2 on CIFAR-10. |

As illustrated in **Figures 9(a)** and **9(b)**, both models show stable training behavior with small generalization gaps. In contrast to CIFAR-100, overfitting is significantly reduced, even under full model fine-tuning. This reflects the lower task complexity and improved class separability of CIFAR-10.

---
#### Qualitative Evaluation

| ResNet-18                                                                     | MobileNetV2                                                         |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| ![](figures/cifar_10_resnet18/wrong_correct_labeled_images.png)               | ![](figures/cifar_10_mobilenet_v2/wrong_clasified_frog.png)         |
| **Figure 10(a):** Correct and incorrect predictions by ResNet-18 on CIFAR-10. | **Figure 10(b):** Incorrect predictions by MobileNetV2 on CIFAR-10. |

Qualitative results in **Figures 10(a)** and **10(b)** indicate that both models correctly classify samples with clear visual features. Misclassifications primarily occur for visually ambiguous images or atypical viewpoints. Compared to CIFAR-100, errors are less frequent and often involve fine-grained confusion rather than broad semantic mistakes.

---
#### Error Analysis: Class Confusion

| ResNet-18                                                                          | MobileNetV2                                                                          |
| ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| ![](figures/cifar_10_resnet18/wrong_frog_classified.png)                           | ![](figures/cifar_10_mobilenet_v2/wrong_clasified_frog.png)                          |
| **Figure 11(a):** Misclassification patterns for the *frog* class using ResNet-18. | **Figure 11(b):** Misclassification patterns for the *frog* class using MobileNetV2. |

As shown in **Figures 11(a)** and **11(b)**, misclassifications for the *frog* class are limited and primarily involve visually similar categories such as *cat*, *deer*, or *truck*. Compared to CIFAR-100, confusion remains tightly constrained within a small subset of related classes, indicating stronger learned representations.

| ResNet-18                                                                       | MobileNetV2                                                                                |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| ![](figures/cifar_10_resnet18/class_frog_resnet18.png)                          | ![](figures/cifar_10_mobilenet_v2/classes_mobilenetv2.png)                                 |
| **Figure 12(a):** Prediction distribution for the *frog* class using ResNet-18. | **Figure 12(b):** Per-class precision, accuracy, and F1-score for MobileNetV2 on CIFAR-10. |

In **Figure 12(a)** we can see that ResNet-18 correctly classifies the *frog* class in the vast majority of cases, with only a small number of misclassifications distributed across visually similar categories such as *cat*, *deer*, *bird*, and *truck*. The dominance of the correct class indicates strong class separation and robust feature learning.

In contrast, **Figure 12(b)** highlights the overall per-class performance of MobileNetV2. While most classes achieve high precision and F1-scores, visually diverse categories such as *cat* and *deer* exhibit slightly reduced performance. Compared to ResNet-18, MobileNetV2 shows marginally weaker discrimination for fine-grained animal classes, consistent with its reduced representational capacity.

### Comparison between CIFAR-10 and CIFAR-100: 

Even though we used the same training range (5 epochs for head training, 5 epochs for last-block fine-tuning, and 15 epochs for full fine-tuning) for both datasets to keep the comparison fair, this range is more suitable for CIFAR-10. For CIFAR-100, the higher class complexity and reduced per-class data mean the model likely needs more epochs especially for the last-block fine tuning phase. We notice in fig2a and fig2b, the validation accuracy is still increasing for both architectures. So with the same 5-5-15 setup it is partially under-trained, which naturally results in lower final accuracy for the CIFAR-100.
