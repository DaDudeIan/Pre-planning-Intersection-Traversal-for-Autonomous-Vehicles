#import "../../../lib/mod.typ": *
== The Models <c4:model>

With all the previous sections covered, the next step is to define and discuss the models that will be used to predict the path through intersection. The goal is to create and test various models that do this task well, evaluating their performance and comparing them to each other. The models used are commonly used as the back-bone of other, larger models designed for specific tasks. This approach was chosen as it may provide a better understanding of what kind of backbone for a model might yield the greatest results in the context of path-planning. 

Chosen for this project are two groups of models, those that are based on convolution and those that are based on transformers. This is done to answer #RQ(1). The specific models chosen, as presented in detail in the following sections, are the convolution-based U-Net and DeepLabV3+ models, and the transformer-based Vision Transformer and Swin Transformer models. These models were chosen largely due to their popularity within the field of computer vision, and their ability to perform well on a variety of tasks. Comparing these distinct groups of models, will also highlight the differences between them, and how they perform on the task at hand. Additionally, these models were chosen in their bare-bone state, as hardware limitations hindered the usage of vast and complex networks.

Furthermore, it should be noted that other DL methodologies were considered. First, is reinforcement learning, which was not chosen as it would add unnecessary complexity to the project. Second, is the use of generative models, which were also not chosen as they introduce a massive change in training algorithm. Both of these points highlight the fact that their implementation would require a significant paradigm shift in this project, which was deemed to go beyond the scope of this thesis. 

The following sections will present the chosen model, starting with the convolution-based models, followed by the transformer-based models. First, the U-Net model proposed in 2015 @unet_og will be presented, focusing on its reliance of classical convolution and usage of alternative convolutional methods. DeepLab was originally proposed in 2016 @deeplabv1 with the implemented V3+ version introduced in 2018 @deeplabv3p. With this model, the focus will be on the use of atrous convolution and the ASPP module. The first of the transformer-based models is the Vision Transformer (ViT), presented in @c2s2.2.1:cv, which introduced the transformer architecture to the field of computer vision. Finally, the Swin Transformer is a hierarchical transformer model that introduced shifted windows to the transformer architecture, allowing for a more efficient computation of self-attention.

// ResNet-based CNN + U-Net
// Diffusion-based models
// Fully convolutional networks
// Vision Transformers

#include "models/unet.typ"

#include "models/deeplab.typ"

#include "models/vit.typ"

#include "models/swin.typ"


// Other considerations:
// - RL (no, because of added complexity)
// While reinforcement learning has its merits in sequential decision-making tasks, it typically requires a different setup—defining a reward function, handling exploration, and dealing with potentially unstable training dynamics. For generating paths (which is more of a structured prediction or segmentation problem), RL might add unnecessary complexity without clear benefits over the more straightforward supervised or generative approaches.