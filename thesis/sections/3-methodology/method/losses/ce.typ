#import "../../../../lib/mod.typ": *

=== Cross-Entropy Loss   <c4:cross-entropy>

Cross-Entropy loss is the main driver behind the models' ability to ascribe every pixel to exactly one semantic class---background, left, right, ahead, or layered. Whereas its binary counterpart BCE, explained in @c4:bce_loss, concerns itself with two mutually-exclusive classes, CE generalizes the idea to multiple classes. Formally, whereas BCE is $C in NN_2$, CE encompasses all classes $C in NN_(>2)$. Thus, CE suits the task of multi-class segmentation, where each pixel belongs to one of $C$ classes, perfectly. 

CE measures the dissimilarity between the ground truth label distribution and the model's predicted label distribution. The loss is defined as the negative log-likelihood of the true class label given the predicted class probabilities. The definition of CE is as follows:

Let 
$
  bold(p)_n = [p_(n,1), ..., p_(n,C)], #h(1cm) bold(y)_n = [y_(n,1), ..., y_(n,C)], #h(1cm) n in {1, ..., N}
$
denote, respectively, the softmax-normalised class-probabilities returned by the network and the one-hot encoded ground-truth for the $n$-th pixel. To further combat class imbalance, a class-weight vector $bold(w) in RR^C$ is introduced, resulting in the per-sample loss:
$
  cal(L)^((n))_"CE" = - sum_(c=1)^C w_c y_(n,c) log(p_(n,c))
$ <eq:ce_loss>

And aggregating over a batch of $N$ samples, the final mean loss is given by:
$
  cal(L)_"CE" = 1/N sum_(n=1)^N cal(L)^((n))_"CE"
$
For this project, there is a very heavy weight imbalance between the classes, with the background class being the most frequent. To counter this, a class-weight vector $bold(w)$ is introduced, which is inversely proportional to the frequency of each class in the training set. Empirically, roughly 94% of all pixels belong to the background class, while the remaining 6% are split between the other four classes. The class-weight vector is defined as follows:
#std-block(
    listing(
        ```python
        class_counts = torch.tensor([152_000, 2_000, 2_000, 2_000, 2_000])
        weights = torch.log1p(class_counts.sum() / class_counts)
        ```,
        caption: [Weight vector definition.]
    )
)
which, mathematically, is
$
  w_c = log(1 + (sum_k f_k)/(f_c))
$ <eq:loss_weights>
where $f_c$ is the class frequency and $f_k$ is the number of pixels for each class. A naïve normalization such as $w_c = 1 slash f_c$ would inflate the smaller classes' weights by orders of magnitude and cause unstable training. Specifically, it may inflate gradients by as much as $f_max slash f_min approx 76$. Therefore, the implementation uses log-compressed inverse-frequency as shown. Equation @eq:loss_weights preserves the desired ordering of $w_c #h(0mm) arrow.t$ as $f_c #h(0mm) arrow.b$, i.e. the weights are inversely proportional to the class frequencies, but only grows $cal(O)(log f_c^(-1))$ instead of $cal(O)(f_c^(-1))$. This also caps the largest weight to $log(1+5) approx 1.79$ rather than 5, preventing exploding gradients.

Finally, to update the model weights during training, the gradient is found by differentiating @eq:ce_loss which gives
$
    (partial cal(L)_"CE")/(partial z_(n,c)) = w_c (p_(n,c) - y_(n,c))
$
where $z_(n,c)$ is the output logits from the models.

Alternative methods were considered for handling the semantic classification. Dice similarity, focal loss, and Kullback-Leibler divergence were considered but ultimately rejected as the primary classification loss function. Dice struggles with extreme class imbalance unless carefully tuned, while focal loss introduces an additional focusing parameter $gamma$ whose optimal value varies per dataset. KL divergence is not suitable for multi-class classification as it assumes a Gaussian distribution, which is not the case here. Thus CE was chosen as the cornerstone for the losses around which the more bespoke losses are built.

These bespoke losses are introduced in the following sections, meant to work hand-in-hand with the CE loss introduced here. CE handles semantic correctness really well, nudging each pixel towards their categorical label. This is however also its great limitation, and is therefore combined with the following sections' losses. First, the cold map loss adds spatial prior knowledge to the model, and the continuity/PH loss injects topological constraints.