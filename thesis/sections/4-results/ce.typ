#import "../../lib/mod.typ": *

#let base = "../../figures/img/results"

// =============== Deeplab ================
#let deeplab_ce_e10_test1 = image(base+"/deeplab/deeplab_ce_e10_test1.png")
#let deeplab_ce_e10_test2 = image(base+"/deeplab/deeplab_ce_e10_test2.png")
#let deeplab_ce_e10_train = image(base+"/deeplab/deeplab_ce_e10_train1.png")

#let deeplab_ce_e100_test1 = image(base+"/deeplab/deeplab_ce_e100_test1.png")
#let deeplab_ce_e100_test2 = image(base+"/deeplab/deeplab_ce_e100_test2.png")
#let deeplab_ce_e100_train = image(base+"/deeplab/deeplab_ce_e100_train1.png")

#let deeplab_ce_e300_test1 = image(base+"/deeplab/deeplab_ce_e300_test1.png")
#let deeplab_ce_e300_test2 = image(base+"/deeplab/deeplab_ce_e300_test2.png")
#let deeplab_ce_e300_train = image(base+"/deeplab/deeplab_ce_e300_train1.png")

#let deeplab_ce_test_graph = image(base+"/deeplab/deeplab_ce_test_graph.png")
#let deeplab_ce_train_graph = image(base+"/deeplab/deeplab_ce_train_graph.png")

// =============== U-Net ================
#let unet_ce_e10_test1 = image(base+"/unet/unet_ce_e10_test1.png")
#let unet_ce_e10_test2 = image(base+"/unet/unet_ce_e10_test2.png")
#let unet_ce_e10_train = image(base+"/unet/unet_ce_e10_train1.png")

#let unet_ce_e100_test1 = image(base+"/unet/unet_ce_e100_test1.png")
#let unet_ce_e100_test2 = image(base+"/unet/unet_ce_e100_test2.png")
#let unet_ce_e100_train = image(base+"/unet/unet_ce_e100_train1.png")

#let unet_ce_e300_test1 = image(base+"/unet/unet_ce_e300_test1.png")
#let unet_ce_e300_test2 = image(base+"/unet/unet_ce_e300_test2.png")
#let unet_ce_e300_train = image(base+"/unet/unet_ce_e300_train1.png")

#let unet_ce_test_graph = image(base+"/unet/unet_ce_test_graph.png")
#let unet_ce_train_graph = image(base+"/unet/unet_ce_train_graph.png")

// =============== ViT ================
#let vit_ce_e10_test1 = image(base+"/vit/vit_ce_e10_test1.png")
#let vit_ce_e10_test2 = image(base+"/vit/vit_ce_e10_test2.png")
#let vit_ce_e10_train = image(base+"/vit/vit_ce_e10_train1.png")

#let vit_ce_e100_test1 = image(base+"/vit/vit_ce_e100_test1.png")
#let vit_ce_e100_test2 = image(base+"/vit/vit_ce_e100_test2.png")
#let vit_ce_e100_train = image(base+"/vit/vit_ce_e100_train1.png")

#let vit_ce_e300_test1 = image(base+"/vit/vit_ce_e300_test1.png")
#let vit_ce_e300_test2 = image(base+"/vit/vit_ce_e300_test2.png")
#let vit_ce_e300_train = image(base+"/vit/vit_ce_e300_train1.png")

#let vit_ce_test_graph = image(base+"/vit/vit_ce_test_graph.png")
#let vit_ce_train_graph = image(base+"/vit/vit_ce_train_graph.png")

// =============== Swin ================
#let swin_ce_e10_test1 = image(base+"/swin/swin_ce_e10_test1.png")
#let swin_ce_e10_test2 = image(base+"/swin/swin_ce_e10_test2.png")
#let swin_ce_e10_train = image(base+"/swin/swin_ce_e10_train1.png")

#let swin_ce_e100_test1 = image(base+"/swin/swin_ce_e100_test1.png")
#let swin_ce_e100_test2 = image(base+"/swin/swin_ce_e100_test2.png")
#let swin_ce_e100_train = image(base+"/swin/swin_ce_e100_train1.png")

#let swin_ce_e300_test1 = image(base+"/swin/swin_ce_e300_test1.png")
#let swin_ce_e300_test2 = image(base+"/swin/swin_ce_e300_test2.png")
#let swin_ce_e300_train = image(base+"/swin/swin_ce_e300_train1.png")

#let swin_ce_test_graph = image(base+"/swin/swin_ce_test_graph.png")
#let swin_ce_train_graph = image(base+"/swin/swin_ce_train_graph.png")

== Cross-Entropy Standalone

To create a baseline to which the topology-based loss functions can be compared, the first results to be presented are those from the models trained purely on the cross-entropy loss.

#std-block(breakable: false,
  figure(
    grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: 0mm,
      {deeplab_ce_e10_test1}, {unet_ce_e10_test1}, {vit_ce_e10_test1}, {swin_ce_e10_test1},
      {deeplab_ce_e10_test2}, {unet_ce_e10_test2}, {vit_ce_e10_test2}, {swin_ce_e10_test2},
      [#subfigure("(a)") DeepLabv3+], [#subfigure("(b)") U-Net], [#subfigure("(c)") ViT], [#subfigure("(d)") Swin],
    ),
    caption: [Results after the #nth(10) epoch]
  )
)

#std-block(breakable: false,
  figure(
    grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: 0mm,
      {deeplab_ce_e100_test1}, {unet_ce_e100_test1}, {vit_ce_e100_test1}, {swin_ce_e100_test1},
      {deeplab_ce_e100_test2}, {unet_ce_e100_test2}, {vit_ce_e100_test2}, {swin_ce_e100_test2},
      [#subfigure("(a)") DeepLabv3+], [#subfigure("(b)") U-Net], [#subfigure("(c)") ViT], [#subfigure("(d)") Swin],
    ),
    caption: [Results after the #nth(100) epoch]
  )
)

#std-block(breakable: false,
  figure(
    grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: 0mm,
      {deeplab_ce_e300_test1}, {unet_ce_e300_test1}, {vit_ce_e300_test1}, {swin_ce_e300_test1},
      {deeplab_ce_e300_test2}, {unet_ce_e300_test2}, {vit_ce_e300_test2}, {swin_ce_e300_test2},
      [#subfigure("(a)") DeepLabv3+], [#subfigure("(b)") U-Net], [#subfigure("(c)") ViT], [#subfigure("(d)") Swin],
    ),
    caption: [Results after the #nth(300) epoch]
  )
)


=== Training and Validation Graphs

#std-block(breakable: false,
  figure(
    grid(
      columns: (1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: (0mm, 1mm, 0mm, 1mm),
      {deeplab_ce_train_graph}, 
      {deeplab_ce_test_graph}, 
      grid.cell([#subfigure("(a)") DeepLabv3+], colspan: 2),
      {unet_ce_train_graph}, 
      {unet_ce_test_graph}, 
      grid.cell([#subfigure("(b)") U-Net], colspan: 2),
      {vit_ce_train_graph}, 
      {vit_ce_test_graph}, 
      grid.cell([#subfigure("(c)") ViT], colspan: 2),
      {swin_ce_train_graph},
      {swin_ce_test_graph},
      grid.cell([#subfigure("(d)") Swin], colspan: 2),
    ),
    caption: [Results after the #nth(300) epoch]
  )
)

=== Test on Training Set
#std-block(breakable: false,
  figure(
    grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: 0mm,
      {deeplab_ce_e10_train}, {unet_ce_e10_train}, {vit_ce_e10_train}, {swin_ce_e10_train},
      [#subfigure("(a)") DeepLabv3+], [#subfigure("(b)") U-Net], [#subfigure("(c)") ViT], [#subfigure("(d)") Swin],
    ),
    caption: [Results after the #nth(10) epoch]
  )
)

#std-block(breakable: false,
  figure(
    grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: 0mm,
      {deeplab_ce_e100_train}, {unet_ce_e100_train}, {vit_ce_e100_train}, {swin_ce_e100_train},
      [#subfigure("(a)") DeepLabv3+], [#subfigure("(b)") U-Net], [#subfigure("(c)") ViT], [#subfigure("(d)") Swin],
    ),
    caption: [Results after the #nth(100) epoch]
  )
)

#std-block(breakable: false,
  figure(
    grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: 0mm,
      {deeplab_ce_e300_train}, {unet_ce_e300_train}, {vit_ce_e300_train}, {swin_ce_e300_train},
      [#subfigure("(a)") DeepLabv3+], [#subfigure("(b)") U-Net], [#subfigure("(c)") ViT], [#subfigure("(d)") Swin],
    ),
    caption: [Results after the #nth(300) epoch]
  )
)