#import "../../../lib/mod.typ": *
== The Models <c4:model>

With all the previous sections covered, the next step is to define and discuss the models that will be used to predict the path through intersection. The goal is to create and test various models that do this task well, evaluating their performance and comparing them to each other. The models used are the very bare-bones version of the models, with little to no modifications applied to them. This approach was chosen as it may provide a better understanding of what kind of backbone for a model might yield the greatest results in the context of path-planning. 

// ResNet-based CNN + U-Net
// Diffusion-based models
// Fully convolutional networks
// Vision Transformers

#include "models/unet.typ"




// Other considerations:
// - RL (no, because of added complexity)
// While reinforcement learning has its merits in sequential decision-making tasks, it typically requires a different setup—defining a reward function, handling exploration, and dealing with potentially unstable training dynamics. For generating paths (which is more of a structured prediction or segmentation problem), RL might add unnecessary complexity without clear benefits over the more straightforward supervised or generative approaches.