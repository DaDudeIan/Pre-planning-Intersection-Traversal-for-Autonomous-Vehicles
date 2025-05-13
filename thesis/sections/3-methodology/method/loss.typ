#import "../../../lib/mod.typ": *

== Loss Function Design #checked <c4:loss>

// Bare bones loss function: dot product
// 
// class imbalance (a lot more background pixels than path pixels)
// - Binary Cross Entropy (BCE). Binary segmentation, pixel-wise classification.
// - Dice loss. Overlap between the predicted and ground truth masks. (https://paperswithcode.com/paper/generalised-dice-overlap-as-a-deep-learning)
// - IoU loss. Intersection over Union. Differential approximation (soft IoU loss)
// - Focal loss.
// 
// Distance based
// - Hausdorff distance. Maximum distance between two sets. Penalizes outliers.
// - Chamfer distance. Average distance between two sets.
// 
// Smoothness
// - Heavily penalize breaks. Topology analysis.
// - Total Variation (TV) loss. Penalize sharp changes.
// 
// Dot product
// + BCE or Dice
// + Distance based (Hausdorff, Chamfer)
// + Topology/Smoothness
// 
// For diffusion: combine with standard diffusion loss (MSE)

One of the most critical parts of designing a deep learning model, is the creation of the loss function that will guide the training. The loss function is a measure of how well a model is performing, and it is used to adjust the model's parameters during training. Therefore, the choice of loss function is crucial to the success of the model. In this section, I will discuss the design of the loss functions used to train the selected models. It will consist of a combination of different loss functions, each designed to capture different aspects of the problem at hand. Firstly, I will cover the utilization of a common classification loss function, the #acr("CE") loss, which is used to measure the difference between the predicted and true distributions. CE will be the main driving force behind the models prediction of the correct labels for each path through the intersection. 

Secondly, two different methods for enforcing topological constraints will be presented. First of these is the development of the novel "cold map"-based loss function, which is supposed to guide the model by penalizing points further away from the true path subject to some threshold. In combination with this, I will discuss the use of #acr("BCE") loss. Finally, a topology-based loss function focusing on enforcing specific betti numbers will be presented. 

#include "losses/ce.typ"

#include "losses/cmap.typ"

#include "losses/topo.typ"


