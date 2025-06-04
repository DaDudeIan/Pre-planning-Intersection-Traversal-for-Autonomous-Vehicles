#import "../../../../lib/mod.typ": *
=== Vision Transformer   <c4:vit>

#let linear_proj_colour = rgb("FFCCE6")
#let embedding_colour = rgb("CC99FF")
#let encoder_colour = rgb("E1D5E7")
#let segmentation_head_colour = rgb("FFCE9F")

// Hello #ball(linear_proj_colour) #ball(embedding_colour) #ball(encoder_colour) #ball(segmentation_head_colour)

For this project, two of the most widely used transformer-based models were chosen, the Vision Transformer (ViT) and the Swin Transformer. The ViT was presented in detail in @c2s2.2.1:cv, highlighting it being the first connection between the NLP method called Attention and computer vision.

#let fig1 = { image("../../../../figures/img/models/vit/architecture.png") }

#std-block(breakable: false)[
  #figure( fig1,
  caption: [Vision Transformer.]
) <fig:vit_architecture>
]
The ViT model is a pure transformer model, meaning it does not use any convolutional layers. Instead, it relies on self-attention mechanisms to process the input data. The model consists of several key components, including a linear projection layer #ball(linear_proj_colour), an embedding layer #ball(embedding_colour), an encoder #ball(encoder_colour), and a segmentation head #ball(segmentation_head_colour). The linear projection layer is responsible for transforming the input data into a format suitable for the transformer architecture. The embedding layer then maps the input data into a higher-dimensional space, allowing the model to capture more complex relationships between the input features. The encoder consists of multiple transformer blocks that apply self-attention and feed-forward networks to process the input data. Finally, the segmentation head generates the output predictions based on the processed features.

Much of this functionality is offered by the PyTorch framework, where the ViT model is implemented as a class. The largest difference from the base ViT presented in @c2s2.2.1:cv is the change from the classification head to the segmentation head #ball(segmentation_head_colour). The core of the network therefore remains unchanged: a single-stride convolutional projection #ball(linear_proj_colour) decomposes the $400 times 400$ input image into non-overlapping $16 times 16$ patches ($G = 25$ along each axis), each of which is flattened and linearly embedded #ball(embedding_colour). A learnable class token is prepended so that the model structure remains identical to the original ViT, but it is discarded after the encoder since the downstream task is dense prediction rather than classification. 

The transformer encoder #ball(encoder_colour) is an unmodified stack of twelve blocks with 768 hidden units, 12 self-attention heads, and a 3072-dimensional MLP. A dropout of 0.2 follows the encoder output. The sequence of patch embeddings is then reshaped back to its 2-D layout, yielding a feature map of shape [B, 768, 25, 25]. A lightweight segmentation head #ball(segmentation_head_colour)—implemented as a single $1 times 1$ convolution—projects this map to the required five classes; bilinear up-sampling restores the original spatial resolution.

Training from random initialisation removes any dependence on large-scale external datasets and keeps the experimental comparison fair with respect to U-Net and DeepLabV3+. The trade-off is a significantly longer convergence time and higher memory usage: the quadratic cost of self-attention makes the ViT approximately four times more memory-intensive than U-Net at $400 times 400$ resolution. Even so, the architecture's ability to capture global context is attractive for the intersection-traversal task, where the correct path at one corner may rely on cues many tens of metres away in the satellite frame.

In summary, the ViT configuration employed here can be viewed as the minimal modification of the canonical ViT architecture for dense prediction: keep the patch projection and encoder intact, drop the class token after encoding, and append a shallow $1 times 1$ convolutional head plus up-sampling. Despite the absence of pre-training, this arrangement preserves ViT's long-range modelling capability while producing pixel-aligned outputs suitable for the task at hand. While the ViT offers a compelling baseline for transformer-only segmentation, its flat token structure and quadratic attention cost can become prohibitive at higher resolutions. Recent research therefore turns to hierarchical designs that marry the long-range reasoning of self-attention with the computational efficiency of local windows.