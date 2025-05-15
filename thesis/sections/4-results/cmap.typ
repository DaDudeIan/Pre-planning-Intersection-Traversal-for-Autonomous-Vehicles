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
    caption: [Results after the #nth(10) epoch]
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
    caption: [Results after the #nth(50) epoch]
  )
)

=== Training and Validation Graphs

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
      {deeplab_cmap_e10_train}, {unet_cmap_e10_train}, {vit_cmap_e10_train}, {swin_cmap_e10_train},
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
      {deeplab_cmap_e50_train}, {unet_cmap_e50_train}, {vit_cmap_e50_train}, {swin_cmap_e50_train},
      [#subfigure("(a)") DeepLabv3+], [#subfigure("(b)") U-Net], [#subfigure("(c)") ViT], [#subfigure("(d)") Swin],
    ),
    caption: [Results after the #nth(50) epoch]
  )
)