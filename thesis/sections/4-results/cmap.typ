#import "../../lib/mod.typ": *

#let base = "../../figures/img/results"

// =============== Deeplab ================
#let deeplab_cmap_e10_test1 = image(base+"/deeplab/deeplab_cmap_e10_test1.png")
#let deeplab_cmap_e10_test2 = image(base+"/deeplab/deeplab_cmap_e10_test2.png")
#let deeplab_cmap_e10_train = image(base+"/deeplab/deeplab_cmap_e10_train1.png")

#let deeplab_cmap_e50_test1 = image(base+"/deeplab/deeplab_cmap_e50_test1.png")
#let deeplab_cmap_e50_test2 = image(base+"/deeplab/deeplab_cmap_e50_test2.png")
#let deeplab_cmap_e50_train = image(base+"/deeplab/deeplab_cmap_e50_train1.png")

#let deeplab_cmap_test_graph = image(base+"/deeplab/deeplab_cmap_test_graph.png")
#let deeplab_cmap_train_graph = image(base+"/deeplab/deeplab_cmap_train_graph.png")

// =============== U-Net ================
#let unet_cmap_e10_test1 = image(base+"/unet/unet_cmap_e10_test1.png")
#let unet_cmap_e10_test2 = image(base+"/unet/unet_cmap_e10_test2.png")
#let unet_cmap_e10_train = image(base+"/unet/unet_cmap_e10_train1.png")

#let unet_cmap_e50_test1 = image(base+"/unet/unet_cmap_e50_test1.png")
#let unet_cmap_e50_test2 = image(base+"/unet/unet_cmap_e50_test2.png")
#let unet_cmap_e50_train = image(base+"/unet/unet_cmap_e50_train1.png")

#let unet_cmap_test_graph = image(base+"/unet/unet_cmap_test_graph.png")
#let unet_cmap_train_graph = image(base+"/unet/unet_cmap_train_graph.png")

// =============== ViT ================
#let vit_cmap_e10_test1 = image(base+"/vit/vit_cmap_e10_test1.png")
#let vit_cmap_e10_test2 = image(base+"/vit/vit_cmap_e10_test2.png")
#let vit_cmap_e10_train = image(base+"/vit/vit_cmap_e10_train1.png")

#let vit_cmap_e50_test1 = image(base+"/vit/vit_cmap_e50_test1.png")
#let vit_cmap_e50_test2 = image(base+"/vit/vit_cmap_e50_test2.png")
#let vit_cmap_e50_train = image(base+"/vit/vit_cmap_e50_train1.png")

#let vit_cmap_test_graph = image(base+"/vit/vit_cmap_test_graph.png")
#let vit_cmap_train_graph = image(base+"/vit/vit_cmap_train_graph.png")

// =============== Swin ================
#let swin_cmap_e10_test1 = image(base+"/swin/swin_cmap_e10_test1.png")
#let swin_cmap_e10_test2 = image(base+"/swin/swin_cmap_e10_test2.png")
#let swin_cmap_e10_train = image(base+"/swin/swin_cmap_e10_train1.png")

#let swin_cmap_e50_test1 = image(base+"/swin/swin_cmap_e50_test1.png")
#let swin_cmap_e50_test2 = image(base+"/swin/swin_cmap_e50_test2.png")
#let swin_cmap_e50_train = image(base+"/swin/swin_cmap_e50_train1.png")

#let swin_cmap_test_graph = image(base+"/swin/swin_cmap_test_graph.png")
#let swin_cmap_train_graph = image(base+"/swin/swin_cmap_train_graph.png")

== Cold Map Standalone

This section will serve as a proof-of-concept for the structural prior method presented called cold map loss. The results shown will be from models that have been trained using the cold map loss defined in @c4:cold_loss. The results will be shown for the first 10 epochs and the 50th epoch. The classes shown using the colours of the previous section serve no purpose in these outputs. The cold map loss does not deal with classifying pixels, but purely the structure of the output. Classification of pixels is a crucial part of the task at hand, as it is critical information for an AV to have when approaching an intersection. Therefore, this section will merely show the loss in a proof-of-concept context.



#std-block(breakable: false,
  figure(
    grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: 0mm,
      {deeplab_cmap_e10_test1}, {unet_cmap_e10_test1}, {vit_cmap_e10_test1}, {swin_cmap_e10_test1},
      {deeplab_cmap_e10_test2}, {unet_cmap_e10_test2}, {vit_cmap_e10_test2}, {swin_cmap_e10_test2},
      [#subfigure("(a)") DeepLabv3+], [#subfigure("(b)") U-Net], [#subfigure("(c)") ViT], [#subfigure("(d)") Swin],
    ),
    caption: [Results after the #nth(10) epoch.]
  )
)

#std-block(breakable: false,
  figure(
    grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: 0mm,
      {deeplab_cmap_e50_test1}, {unet_cmap_e50_test1}, {vit_cmap_e50_test1}, {swin_cmap_e50_test1},
      {deeplab_cmap_e50_test2}, {unet_cmap_e50_test2}, {vit_cmap_e50_test2}, {swin_cmap_e50_test2},
      [#subfigure("(a)") DeepLabv3+], [#subfigure("(b)") U-Net], [#subfigure("(c)") ViT], [#subfigure("(d)") Swin],
    ),
    caption: [Results after the #nth(50) epoch.]
  )
)

Already after 10 epochs, the convolutional models are able to produce a reasonable segmentation of the image. Particularly DeepLab is able to produce maps that follow the structure of the road in the image. The paths that it does highlight are fairly wide at this stage, but becomes significantly more refined after the #nth(50) epoch, i.e. they become a fair amount thinner than earlier. U-Net seems less happy with this loss function than DeepLab. After 10 epochs it only learns the straight ahead path somewhat well, with the left and right turns being almost non-existent. The only parts that do exist in these maps are unconnected and appear seemingly randomly, with the only correctness stemming from the fact that pixels occupied by roads are identified. At the 50 epoch mark, it has increased performance slightly, but still does not produce a map that is as good as DeepLab. The outputs are extremely wobbly and is not contained to the correct side of the road.

The transformer-based models do also appear to not be particularly happy with this loss function. ViT is generally able to learn the structure of the road, having only a few blotches off of the road after 50 epochs. It does, however, not appear to be motivated by the loss to create a map that is coherent across te entire image, failing to follow the road in any meaningful way. The splits in the generated blobs, do not seem to follow any apparent structure or pattern, meaning that the network does not understand the underlying structure of roads that are not perfectly perpendicular. Swin does initially show some promise with this loss, as the more simple intersection is selected with very high precision after 10 epochs. This does not improve after the 50th epoch. Like the ViT, it does not seem to be able to learn the structure of a road, as it doesn't highlight the parts of the road that are similarly coloured to some of the surroundings.

=== Training and Validation Graphs

Compared to the purely CE-trained models, the graphs for these models are significantly different. The losses plateau after just a few epochs, with only marginal improvements after 50 epochs. This implies that the core structure of the road is learned very quickly. Furthermore, it implies that the models quickly converge on the structural features defined by the cold maps from the dataset.


#std-block(breakable: false,
  figure(
    grid(
      columns: (1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: (0mm, 1mm, 0mm, 1mm),
      {deeplab_cmap_train_graph}, 
      {deeplab_cmap_test_graph}, 
      grid.cell([#subfigure("(a)") DeepLabv3+], colspan: 2),
      {unet_cmap_train_graph}, 
      {unet_cmap_test_graph}, 
      grid.cell([#subfigure("(b)") U-Net], colspan: 2),
      {vit_cmap_train_graph}, 
      {vit_cmap_test_graph}, 
      grid.cell([#subfigure("(c)") ViT], colspan: 2),
      {swin_cmap_train_graph},
      {swin_cmap_test_graph},
      grid.cell([#subfigure("(d)") Swin], colspan: 2),
    ),
    caption: [Training and Validation graphs for the models trained with the cold map loss. The left column shows the training/test loss and the right column shows the training/validation accuracy. ]
  )
)

Unlike the overfitting seen in the CE-trained models, these models show little to no overfitting. The cold map loss seems to motivate for a more consistent behaviour in the models, hinting at them being able to generalize better to unseen data. This is more is like what is expected of a supervised learning task, where the model is able to learn the structure of the data and generalize to unseen data. Therefore, this loss function seems to be a good candidate to work in conjunction with the CE loss function.

The accuracy graphs largely mirror the behaviour shown in the loss graphs. For some models, such as DeepLabv3+ and ViT, the validation accuracy displays more fluctuations and a decreasing value. These variations could potentially indicate sensitivity to learning rate adjustments from the cosine annealing scheduler, although the inverted spikes apparent in the CE-trained models are less evident here. Their categorical counterparts, U-Net and Swin, exhibit a more stable, increasing accuracy trend, suggesting a more consistent learning process. The fact that models trained on this loss seem to generalize well, shows promise for the use of this loss function in conjunction with the CE loss function, although its seemingly random pixel classification, might trip up the CE loss.

// === Test on Training Set

// #std-block(breakable: false,
//   figure(
//     grid(
//       columns: (1fr, 1fr, 1fr, 1fr),
//       column-gutter: 0mm,
//       row-gutter: 0mm,
//       {deeplab_cmap_e10_train}, {unet_cmap_e10_train}, {vit_cmap_e10_train}, {swin_cmap_e10_train},
//       [#subfigure("(a)") DeepLabv3+], [#subfigure("(b)") U-Net], [#subfigure("(c)") ViT], [#subfigure("(d)") Swin],
//     ),
//     caption: [Results after the #nth(10) epoch]
//   )
// )

// #std-block(breakable: false,
//   figure(
//     grid(
//       columns: (1fr, 1fr, 1fr, 1fr),
//       column-gutter: 0mm,
//       row-gutter: 0mm,
//       {deeplab_cmap_e50_train}, {unet_cmap_e50_train}, {vit_cmap_e50_train}, {swin_cmap_e50_train},
//       [#subfigure("(a)") DeepLabv3+], [#subfigure("(b)") U-Net], [#subfigure("(c)") ViT], [#subfigure("(d)") Swin],
//     ),
//     caption: [Results after the #nth(50) epoch]
//   )
// )