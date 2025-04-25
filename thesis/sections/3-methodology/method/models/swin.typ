#import "../../../../lib/mod.typ": *
=== Swin Transformer <c4:swin>

#let patch_colour = rgb("2226FF")
#let window_colour = rgb("B85450")
#let layer_norm_colour = rgb("D5E8D4")
#let msa_colour = rgb("F8CECC")
#let mlp_colour = rgb("D4E1F5")


// Hello #ball(window_colour) #ball(patch_colour) #ball(layer_norm_colour) #ball(msa_colour)
The Swin Transformer extends the pure-attention idea of ViT with a hierarchical, window-based design that scales gracefully to high-resolution images. As reviewed in §2.2.1, Swin partitions the feature map into fixed-size, non-overlapping windows #ball(window_colour) and computes self-attention only within each window. @fig:swin_sw#subfigure("a") visualises this arrangement for a $2 times 2$ window composed of $4 times 4$ patches #ball(patch_colour); the quadratic cost of attention is now bound by the window area rather than the entire image. @fig:swin_sw#subfigure("b") shows two successive Swin transformer blocks. These ensure the exchange of information across window borders, where the second block #ball(msa_colour) in a pair shifts the window grid by half the window size, so that the pixels sitting on window edges in the first block #ball(msa_colour) lie at the centre of a window in the next. This “shifted window” scheme allows cross-window interactions with only a negligible increase in computation, while still preserving the locality that makes the model efficient.

#let fig1 = {image("../../../../figures/img/models/swin/sw.png")}
#let fig2 = {image("../../../../figures/img/models/swin/swin_block.png")}

#std-block(breakable: false)[
  #figure( 
    grid(columns: (3fr, 1fr), column-gutter: 5mm,
      fig1, fig2, [#subfigure("(a)") Shifted Window.], [#subfigure("(b)") Two Successive Swin Transformer Blocks.]
    ),
    caption: [Swin Transformer components.]
  ) <fig:swin_sw>
]

A single Swin transformer block, shown in @fig:swin_sw#subfigure("b"), follows the canonical transformer ordering of LayerNorm #ball(layer_norm_colour), multi-head self-attention #ball(msa_colour), and a feed-forward network #ball(mlp_colour), but with two key modifications. First, the attention module is either a standard Window-MSA (W-MSA) or its shifted counterpart (SW-MSA), depicted by the #h(1mm) #ball(msa_colour)-coloured block. Second, the feed-forward network is implemented as a two-layer MLP #ball(mlp_colour) with a GELU activation between the layers. Gaussian Error Linear Units (GELU) is an activation function akin to RELU, but has a Gaussian distribution function, resulting in what is considered to be a smooth version of RELU. Residual connections wrap both the attention and MLP sub-modules, enabling stable training of deep hierarchies.

The original Swin architecture is built from four such stages, each consisting of an alternating sequence of W-MSA and SW-MSA blocks. Between stages, a patch-merging layer concatenates neighbouring $2 times 2$ tokens and projects them to twice the channel dimension, halving both spatial resolution and token count while increasing representational capacity. The resulting $H/4, H/8, H/16, H/32$ feature pyramid closely mirrors the spatial hierarchy of convolutional backbones and therefore integrates naturally with encoder-decoder segmentation heads.

The implementation employed in this work instantiates the `swin_base_patch4_window7_224` variant through `timm`, but with three modifications to suit the task. First, the image size is set to $400 times 400$ so that the initial $4 times 4$ patch projection still yields a token grid divisible by the $7 times 7$ window. Second, the model is created with features_only=True, exposing the output of each stage; however, only the final stage is used for the present, lightweight decoder. Third, a small segmentation head replaces the classification head: a $3 times 3$ convolution reduces the 1024-channel backbone output to 256, a ReLU introduces non-linearity, and a $1 times 1$ projection produces the five class logits, which are finally up-sampled bilinearly to the original resolution.