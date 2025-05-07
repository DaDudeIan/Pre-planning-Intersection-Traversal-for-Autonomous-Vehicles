#import "../../../../lib/mod.typ": *

=== Cold Map Loss <c4:cold_loss>

The cold map loss is a novel loss function designed to enforce topological constraints on the predicted path. It is based on the idea of using a cold map, which is a grid of the same size as the input image, where the intensity of each cell is a value derived from the distance to the nearest path pixel. Before covering them and their creation, the BCE loss function is presented, as it is closely related the CE loss but for binary classification tasks, such as when looking at the structure of the predicted path.

==== Binary Cross-Entropy Loss #checked <c4:bce_loss>

The #acr("BCE") loss is a commonly used loss function for binary segmentation tasks, which is relevant for the task at hand due to the pixel-subset nature of the problem, i.e. pixels can be either zero or non-zero. Furthermore, it is well-versed in handling heavily imbalanced data, which is the case when dealing with classification tasks where one class is much more prevalent than the other, like path and non-path pixels in an image. These properties make it ideal for this problem, as the background pixels are much more prevalent than the path pixels. The implementation is using the definition from PyTorch #footnote([Full implementation details can be found in the official documentation: https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html]), which is defined as follows:


$
  cal("l")(x,y) = L = {l_1, dots , l_N}^T
$ <eq:bce_loss>
with
$
  l_n = - w_n [y_n log x_n + (1 - y_n) log(1 - x_n)]
$ <eq:bce_loss_n>
where $w_n$ is a weight parameter, $y_n$ is the ground truth label, $x_n$ is the predicted label, and $cal("l")$ is subject to
$
  cal("l")(x,y) = cases(
    #align(right)[#box[$"mean"(L)$], #h(5mm)]& "if reduction = 'mean'",
    #align(right)[#box[$"sum"(L)$],]& "if reduction = 'sum'"
  )
$ <eq:bce_loss_reduction>
depending on the reduction parameter. Looking at the inner bracket @eq:bce_loss_n, the left-hand side is activated when the ground truth label is $1$. It evaluates how well the positive class's predicted probability $x_n$ aligns with the ground truth. A smaller loss is achieved when the predicted probability is close to $1$. The right-hand side of @eq:bce_loss_n is activated when the ground truth label is $0$. It evaluates how well the negative class's predicted probability $x_n$ aligns with the ground truth. A smaller loss is achieved when the predicted probability is close to $0$. The BCE loss is then calculated as the sum or mean of the individual losses, depending on the `reduction` parameter. The weight parameter $w_n$ can be used to scale the output of the loss function, which is key in making the cold map loss work. 

Formally, #acr("BCE") quantifies the dissimilarity between the predicted probabilities and the actual labels, giving a sense of how well the model is performing. For example, calculating the BCE with $y=1$ and $x=0.8$ gives

#let nonum(eq) = math.equation(block: true, numbering: none, eq)
#nonum($-1 dot (1 dot log(0.8) + (1 - 1) dot log(1 - 0.8)) = -log(0.8) = 0.223$)

This value represents the dissimilarity between the predicted probability of $0.8$ and the actual label of $1$. A lower value indicates that the model's prediction is closer to the ground truth, suggesting a more accurate classification. Alternatively, the value can be near $1$, indicating a large discrepancy between the predicted and actual labels. So, a value of $0.223$ is a good result, as it indicates that the model is performing well, with some room for improvement. For contrast, if the predicted label is $0.4$, but the true label is $1$, the dissimilarity would be $-log(0.4) approx 0.916$. This higher value reflects a more significant error in prediction. From this, note that the function calculates the dissimilarity for both positive and negative classes. So, in the case of the predicted label being $0.4$ and the true label being $0$, the loss value would be much better at just $-log(1-0.4) approx 0.51$. 

// In the PyTorch implementation, the BCE loss can be either summed or averaged over the batch, depending on the `reduction` parameter. While using the sum method can provide stronger signals on rare but critical pixels, averaging might help maintain stability across batches, especially when dealing with very imbalanced datasets. Thus, the mean method is chosen for this project, as it leads to a more stable training process with smaller fluctuations in the loss values. The BCE loss is calculated using the following code snippet:

// #listing([
//   ```python
// def bce_loss_torch(path_gt, path_pred, reduction) -> torch.Tensor:
//     bce = torch.nn.BCELoss(reduction=reduction)

//     bce_loss = bce(path_pred, path_gt)
    
//     return bce_loss

// class BCELoss(nn.Module):
//   def __init__(self, weight: float = 1.0):
//     super(BCELoss, self).__init__()
//     self.weight = weight
      
//   def forward(self, path_gt: torch.Tensor, path_pred: torch.Tensor, reduction: str = 'mean'):
//     loss = bce_loss_torch(path_gt, path_pred, reduction)
//     return self.weight * loss
//   ```
// ],
// caption: [Implementation of the #acr("BCE") loss function using PyTorch.]
// ) <code.bce_loss>

// To implement the BCE loss function, a smaller wrapper class for the existing PyTorch implementation is created. The class `BCELoss` inherits from `torch.nn.Module` and has a single parameter, `weight`, which is used to scale the loss value. The `forward` method takes the ground truth path `path_gt`, the predicted path `path_pred`, and the reduction method as inputs. The method then calculates the loss using the `bce_loss_torch` function and scales it by the `weight` parameter. The function `bce_loss` calculates the BCE loss between the ground truth and predicted paths. It takes the paths to the ground truth and predicted images as input, reads them using OpenCV, and converts them to PyTorch tensors after normalization, since the images are stored as 8-bit greyscale images. The function creates a BCE loss `criterion` using `torch.nn.BCELoss` and calculates the loss using the ground truth and predicted tensors. The loss value is then returned as a float. In the tensor version of the function, the paths are already tensors, and the function can be called directly with the tensors as input.

// #std-block(breakable: false)[
//   #v(-1em)
//   #box(
//     fill: theme.sapphire,
//     outset: 0em,
//     inset: 0em,
//   )
//   #figure(
//   image("../../../../figures/img/loss_example/cmap_loss_comparison4.png", width: 100%),
//   caption: [The ground truth #ball("#E4B96D") compared to some drawn path #ball(white). The losses above the plots are the BCE loss using function in @code.bce_loss using `mean` as the string passed as the reduction method. ]
// ) <fig:bce_loss_comp>
// ]

// Examples of the `mean`-based BCE loss is shown in @fig:bce_loss_comp, showing various interesting aspects of the loss function. (b) shows an expectedly high loss value, as the path is far from the true path. This matches the case for the cold map loss. (a), (c), and (d) show very similar loss values, despite being vastly different, both in terms of closeness to the path, but also where they are going to and from, further highlighting the need for a topology-based loss. Even their sum counterparts show very similar values. Interestingly, the losses seen in (f) and (h) are very different, when the cold map based loss showed them as being equal. This shows that BCE is more sensitive to the topology of the path, but as (g) shows, it still gives a very low loss value when a path has branches. All of this shows that some topology analysis is needed for the loss function to capture realistic paths.

Other considerations for handling imbalanced data include methods like Dice @dice_loss similarity coefficient and Focal loss @focal_loss, which can also be effective in certain contexts. The Dice similarity coefficient is a measure of overlap between two samples, and is particularly useful when dealing with imbalanced data. The Focal loss is designed to address the class imbalance problem by focusing on hard examples that are misclassified. These methods can be used in conjunction with the BCE loss to improve the model's performance, especially when dealing with heavily imbalanced datasets. But for the purposes of this project, the BCE loss is expected to be sufficient.

==== Cold Maps

//#text("REMEMBER: Cold and heat map used to penalize false positives and false negatives, respectively. Comment on trivial optimum (optimizer just driving everything to 0.", fill: red)

The cold map loss is the first of two loss functions that will improve topological soundness within the output from the models. This involves using the predicted path generated by the model, and comparing it to the cold map from the dataset. The creation of the cold maps are detailed in @c4:cold_maps. Briefly, the cold maps are grids of the same size as the input image, where the intensity of each cell is a value derived from the distance to the nearest path pixel magnified beyond some threshold. 

The main idea behind the cold map loss is to introduce spatial penalty that increases as the distance from the true path increases. Though it is similar to #acr("BCE"), it differs in some key aspects. It is not pixel-wise, but rather a global loss that is calculated over the entire image. This means that slight deviations from the true path are penalized less than those with a larger discrepancy; BCE simply checks for classes. This property of the loss function is a desirable trait for path-planning tasks, as minor offsets from the true path are less critical than larger ones. This loss function is defined as follows:

$
  cal("L") = sum_(i=1)^H sum_(j=1)^W C_(i j) P_(i j)
$ <eq:loss_cold>

where $C_(i j)$ is the cold map value at pixel $(i, j)$, and $P_(i j)$ is the predicted path value at pixel $(i, j)$. This version of the loss function is a simple dot product between the cold map and the predicted path. Thus, after flattening the cold map and the predicted path matrices, the loss is calculated as the dot product between the two vectors: 

$
  cal("L")_"cold" = C dot P
$ <eq:loss_cold_flat>

where $C$ is the cold map vector and $P$ is the predicted path vector, giving a scalar value, contributing to the total loss. However, in practice, this on its own does not drive an optimizer to optimize the models in any meaningful way. What an optimizer like Adam or SGD does, is to minimize the loss function by adjusting the model's parameters. This cold map loss does well to drive it towards removing activated pixels, but is does this in a very destructive way. By simply pushing every logit from the model towards $-infinity$ it achieves a perfect score: 0. This means that the model will not output any meaningful predictions, as it will simply push all logits to $-infinity$. This is because of the sigmoid function used inside the loss's implementation. Simply doing the aforementioned dot product requires the output of the model to be in the range of $[0, 1]$, which is not the case for the logits, so it is put through a sigmoid function. Thus, as the optimizers tries to lower the loss, it inevitably ends up with $sigma(-infinity) = 0$, which is the perfect score.

To combat this and to make the cold map loss viable, two major changes are made. The equations shown hitherto do work in the sense that they penalize point further away, but they don't reward correct predictions either. This means that the loss penalizes false positives, but not false negatives. Thus, the first of the major changes is introduced. The opposite of the cold map is introduced, i.e. a heat map. This is simply the reversed cold map: $h_"map" = 1 - c_"map"$. This operation essentially invert the cold map, meaning that the penalty is higher the closer the ground truth path it is. This heat map penalizes false negatives, and in combination the two will drive models to output paths that are close to the true path. 

The second major change is the fact that BCE is used with the cold and heat maps as the weights. Since the output from the models are of the same size as the cold map, these maps can be used to individually weight the loss for each pixel. However, simply adding these two together will give a combined map that is simply 1 all over, so a $alpha$ and $beta$ terms are introduced to add more credence to one map than the other. The combined map is thus found by the following equation:
$
  w = alpha h_"map" + beta c_"map"  
$

where $alpha = 2$ and $beta = 1$ because it should reward true positives more than anything else. This, along with the output logits from the model, is passed to the BCE loss function, which will then calculate the loss as a weighted sum of the individual losses. 

//This value does, however, grow extremely quickly, as the dot product, as shown by @eq:loss_cold, is simply a sum over the entire image. This means that the loss value will be very high, even for small deviations from the true path, and extremely high for large deviations or noisy images, as expected from the model during early stages of training. While this rapid growth can be combated by introducing a very low weight to the loss function, doing so would also mean that smaller deviations become irrelevant, which in undesired. Thus, inspired by BCE, I will introduce a mean reduction to the loss function, which will divide the loss by the number of pixels in the image. This will ensure that the loss value is more stable and that the model can learn from smaller deviations. With this, the implementation of a cold map loss, will be based on the following equation:

// The implementation of  is very straightforward, and is shown in the code listing below:

// #listing([
//   ```python
// def cmap_loss(cmap_gt, path_pred, reduction) -> torch.Tensor:
//   cmap_f = torch.flatten(cmap_gt)
//   path_f = torch.flatten(path_pred)
  
//   loss = torch.dot(cmap_f, path_f)
    
//   return loss if reduction != 'mean' else loss / len(cmap_f)

// class CmapLoss(nn.Module):
//   def __init__(self, weight: float = 1.0, reduction: str = 'mean'):
//     super(CmapLoss, self).__init__()
//     self.weight = weight
      
//   def forward(self, cmap_gt: torch.Tensor, path_pred: torch.Tensor):
//     loss = cmap_loss_torch(cmap_gt, path_pred, self.reduction)
//     return self.weight * loss
//   ```
// ],
// caption: [Implementation of the cold map loss calculation using PyTorch.]
// ) <code.cold_loss>

// To implement this loss function, a class `CmapLoss` is created, which inherits from `torch.nn.Module`. The class has a single parameter, `weight`, which is used to scale the loss value. The `forward` method takes the ground truth cold map `cmap_gt` and the predicted path `path_pred` as inputs. The method then calculates the loss using the `cmap_loss` function and scales it by the `weight` parameter. The function `cmap_loss` takes two PyTorch tensors as inputs: `cmap_gt`, which represents the ground truth cold map, and `path_pred`, which is the predicted path output from the model. The cold map and predicted path are initially matrices with dimensions corresponding to the image's height and width. To compute the loss as a single scalar, both matrices are flattened into one-dimensional vectors. The function then calculates the dot product between these two vectors using `torch.dot`, effectively summing the element-wise products. If the `reduction` parameter is set to `mean`, the loss is divided by the number of elements in the vectors, which is the total number of pixels in the image. 

Examples of this loss function in action is shown in @fig:cmap_loss_comp. The left and center top row plots overlays a complete, single width path on top of the cold map. This highlights the fact that paths that are close to the true path, are penalized less and the further away, the more the penalty explodes in value. The last image in the top row shows how the loss handles a noisy image. As it can be seen, the loss is significantly higher than the rest. This is a desired trait of the loss function, as noise is just about the exact opposite of a continuous path. The bottom row shows alternate paths. The path on the rightmost image is the true path, which shows that if the path is dead on, then the penalty is none. The leftmost image, shows a path that is not connected. As shown, it still scores a perfect score. This is because the cold map loss only penalizes the distance from the path, not the topology of the path itself, so breaks will only result in a higher score. Lastly, the center image shows the true path, but with several branches. As seen in the loss value, this also incurs very little penalty. The bottom row of images highlight a dire need for a topology-based loss, which will be explored in @c4:topology_loss. 

#std-block(breakable: false)[
  #figure(
  image("../../../../figures/img/loss_example/cmap_loss_comparison7.png", width: 60%),
  caption: [Paths drawn on top of a cold map with their associated loss. The top row shows fully connected paths, while the bottom row shows paths with breaks and branches, as well as the true path. \ #text("UPDATE FIGURE", fill: red, weight: "black")]
) <fig:cmap_loss_comp>
]

