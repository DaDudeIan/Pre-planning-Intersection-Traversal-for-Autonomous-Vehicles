#import "../../../lib/mod.typ": *
== The Models #checked <c4:model>

With all the previous sections covered, the next step is to define and discuss the models that will be used to predict the path through intersection. The goal is to create and test various models that do this task well, evaluating their performance and comparing them to each other. The models used are commonly used as the back-bone of other, larger models designed for specific tasks. This approach was chosen as it may provide a better understanding of what kind of backbone for a model might yield the greatest results in the context of path-planning. 

Chosen for this project are two groups of models, those that are based on convolution and those that are based on transformers. This is done to answer #RQ(1). The specific models chosen, as presented in detail in the following sections, are the convolution-based U-Net and DeepLabV3+ models, and the transformer-based Vision Transformer and Swin Transformer models. These models were chosen largely due to their popularity within the field of computer vision, and their ability to perform well on a variety of tasks. Comparing these distinct groups of models, will also highlight the differences between them, and how they perform on the task at hand. Additionally, these models were chosen in their bare-bone state, as hardware limitations hindered the usage of vast and complex networks.

Furthermore, it should be noted that other DL methodologies were considered. First, is reinforcement learning, which was not chosen as it would add unnecessary complexity to the project. Second, is the use of generative models, which were also not chosen as they introduce a massive change in training algorithm. Both of these points highlight the fact that their implementation would require a significant paradigm shift in this project, which was deemed to go beyond the scope of this thesis. 

The following sections will present the chosen model, starting with the convolution-based models, followed by the transformer-based models. First, the U-Net model proposed in 2015 @unet_og will be presented, focusing on its reliance of classical convolution and usage of alternative convolutional methods and skip connections. DeepLab was originally proposed in 2016 @deeplabv1 with the implemented V3+ version introduced in 2018 @deeplabv3p. With this model, the focus will be on the use of atrous convolution and the ASPP module. The first of the transformer-based models is the Vision Transformer (ViT), presented in @c2s2.2.1:cv, which introduced the transformer architecture to the field of computer vision. Finally, the Swin Transformer is a hierarchical transformer model that introduced shifted windows to the transformer architecture, allowing for a more efficient computation of self-attention.

// ResNet-based CNN + U-Net
// Diffusion-based models
// Fully convolutional networks
// Vision Transformers

#include "models/unet.typ"

#include "models/deeplab.typ"

#include "models/vit.typ"

#include "models/swin.typ"

Finally, with all the models presented, the next step is to discuss the training strategy used for all models. The training strategy is a crucial part of the model, as it defines how the model will be trained and what kinds of loss functions will be used and how they will be used together. 

=== Training Strategy #checked <c4:training-strategy>
// loss functions (combination, alpha), optimizer (weight decay), scheduler

This section will give a comprehensive overview of the training strategy used for all models. All models will be trained using a combination of loss functions, an optimizer with weight decay, and a learning rate scheduler. Finally, a different number of epochs will be used for each combination of loss functions, as some are simply slower than others, resulting in very long training times.

==== Loss Functions #checked

There are three loss functions that will be used in this project. Cross-entropy (CE) loss is the main driving force behind training the models to correctly label each pixel in the image. CE will be used in combination with both of the topology loss functions separately. The topology loss functions will not trained with the intention of reaching the same level of results, thus it is only the novel cold map based loss that will be used to train the models on its own. This is mainly to show the effectiveness of the loss function in shaping the model's predictions. 

To train with a combination of loss functions, the CE loss will be combined with the topology based ones. It is a common tactic to combine loss functions in order to achieve better results. Loss functions are typically combined by simply adding them together, with a weight for each loss function. This is done to balance the contribution of each loss function to the overall loss. In order to achieve stable training, this weight is typically set such that each loss function contributes equally to the overall loss, i.e. the weights add up to 1. This is achieved by combining the loss functions like this:
$
    cal(L)_"total" = alpha dot cal(L)_1 + (1 - alpha) dot cal(L)_2
$
thus by setting $0 <= alpha <= 1$ the contribution of each loss function can be controlled. For example, if $alpha = 0.5$, then both loss functions contribute equally to the overall loss. And of course if you set $alpha = 1$, then only the first loss function contributes to the overall loss and vice versa if $alpha = 0$. For this project, the value of $alpha$ is chosen dynamically during training. The value for alpha follows the following function:

$
  alpha("epoch") = cases(
    #align(right)[#box[$alpha_"hi"$],]& "if epoch" < T_"warm",
    #align(right)[#box[$alpha_"hi" - (alpha_"hi" - alpha_"lo") dot r$], #h(3mm)]& "if" T_"warm" <= "epoch" < N_"epochs"
  )
$
with $r=("epoch" - T_"warm") / max(1, N_"epochs" - T_"warm")$, where $alpha_"hi"$ and $alpha_"lo"$ are the high and low values for alpha, respectively, $T_"warm"$ is the warm-up period, i.e. the stage where $alpha_"hi"$ is kept as the $alpha$ value, and $N_"epochs"$ is the total number of epochs. Once past $T_"warm"$, the value of $alpha$ will linearly decrease to $alpha_"lo"$, which it will hit at the end of training. Below is a table showing the values for $alpha_"hi"$, $alpha_"lo"$, and $T_"warm"$ for each combination of loss functions, as well as the number of epochs they are trained on:

#let tab = [
  #figure(
    {
      tablec(
        columns: 5,
        alignment: (x, y) => (left, center, center, center, center).at(x),
        header: table.header(
          [], [Cross-entropy], [CE + Cold Map], [CE + Continuity], [Cold Map]
        ),
        [$alpha_"hi"$], [---], [0.9], [0.99],  [---],
        [$alpha_"lo"$], [---], [0.5], [0.5],  [---],
        [$T_"warm"$], [---], [10], [30], [---], 
        [$N_"epochs"$], [300], [100], [100],  [50],
        [$S_"epochs"$], [{10, 20, 50, 100, 300}], [{10, 20, 50, 100}], [{10, 20, 50, 100}], [{10, 20, 50}], []
      )
    },
    caption: [Values for $alpha_"hi"$, $alpha_"lo"$, $T_"warm"$, and $N_"epochs"$ for the different loss function combinations. $S_"epochs"$ is the set of epochs at which the models are checkpointed.]
  )<tab:loss-combinations>
]

#tab

What this means, is that the CE loss will always be the main driving force during training, while topology based losses will gradually become more influential, but never take over more than half of the total loss. The values shown are for all models. The table also shows the number of epochs used for both CE and the cold map loss as standalone. CE on its own is meant to serve as a baseline for the other combined losses. Finally, the epochs at which the models are checkpointed are shown by $S_"epochs"$.

==== Optimizer and Scheduler #checked
The optimizer used for all models is the AdamW optimizer @adamw, which is a variant of the Adam optimizer that includes weight decay. Unlike the standard Adam optimizer, which incorporates L2 regularization by adding a penalty term to the loss function, AdamW decouples weight decay from the gradient-based optimization step. This distinction is important because, in Adam, the interaction between L2 regularization and adaptive learning rates can lead to suboptimal convergence behaviour. In contrast, AdamW applies weight decay directly to the weights during the parameter update step, independently of the gradient computation.

L2 regularization in Adam is implemented top modify the gradient like this:
$
  g_t = nabla f(theta_t) + w_t theta_t
$
where $w_t$ is the regularization coefficient, or rate of decay, and $theta_t$ is the weight at time $t$. This blends the decay term with the gradient, making it sensitive to the optimizer's internal adaptive mechanisms. What AdamW does instead is to adjust the weight decay term to appear in the gradient update step:
$
  theta_(t+1,i) = theta_t,i - eta (1/sqrt(hat(v)_t + epsilon) dot hat(m)_t + w_(t,i)theta_(t,i)), forall t 
$
where $hat(m)_t$ and $hat(v)_t$ are the bias-corrected first and second moment estimates of the gradients, respectively, and $w_(t,i)$ is the weight decay coefficient. This means that the weight decay is applied directly to the weights during the update step. By separating the decay term from the loss gradient, AdamW ensures a more consistent regularization effect and improves generalization. Thus, AdamW is chosen as the optimizer for all models, as it is an enhanced version of Adam that provides better performance in many scenarios.

In combination with the AdamW optimizer, a learning rate scheduler is used to adjust the learning rate during training. The scheduler used is the cosine annealing scheduler, which gradually reduces the learning rate from an initial value to a minimum value over a specified number of epochs before resetting it to the initial value. The cosine annealing scheduler is defined as:
$
  eta_t = eta_min + 1/2 (eta_max-eta_min)(1 + cos(T_"cur"/T_i pi))
$
where $eta_min$ and $eta_max$ are the ranges of the learning rate, with $eta_max$ being the initial learning rate, $T_"cur"$ account for how many epochs have been performed since the last restart, and $T_i$ is the total number of epochs. This form of cosine annealing is known as warm restarts, and is used to improve the performance of the model by allowing it to escape local minima and explore the loss landscape more effectively. This is the main trait desired of the scheduler for this project, as the models should really explore the loss landscape to find the best possible solution. 

Other considerations for the scheduler include exponential decay and plateau decay. Exponential decay is a simple and effective way to reduce the learning rate over time, but it can be too aggressive and lead to premature convergence. Plateau decay, on the other hand, is more adaptive as it lowers the learning rate if the validation loss does not improve for a certain number of epochs. 

With the training strategy now fully defined---including the choice of loss functions, optimizer, and scheduler---the stage is set to evaluate how each model performs under these conditions. The following chapter presents the results of these experiments, comparing both the quantitative metrics and qualitative outputs of each configuration.

// Other considerations:
// - RL (no, because of added complexity)
// While reinforcement learning has its merits in sequential decision-making tasks, it typically requires a different setup—defining a reward function, handling exploration, and dealing with potentially unstable training dynamics. For generating paths (which is more of a structured prediction or segmentation problem), RL might add unnecessary complexity without clear benefits over the more straightforward supervised or generative approaches.