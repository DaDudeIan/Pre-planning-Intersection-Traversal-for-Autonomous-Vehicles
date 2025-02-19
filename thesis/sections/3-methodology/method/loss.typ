#import "../../../lib/mod.typ": *
== Loss Function Design <c4:loss>

// Bare bones loss function: dot product
// 
// class imbalance (a lot more backgrund pixels than path pixels)
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