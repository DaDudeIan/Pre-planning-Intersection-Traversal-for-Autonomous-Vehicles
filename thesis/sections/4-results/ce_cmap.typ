#import "../../lib/mod.typ": *

#let base = "../../figures/img/results"

// =============== Deeplab ================
#let deeplab_ce-cmap_e50_test1 = image(base+"/deeplab/deeplab_ce-cmap_e50_test1.png")
#let deeplab_ce-cmap_e50_test2 = image(base+"/deeplab/deeplab_ce-cmap_e50_test2.png")
#let deeplab_ce-cmap_e50_train = image(base+"/deeplab/deeplab_ce-cmap_e50_train1.png")

#let deeplab_ce-cmap_e100_test1 = image(base+"/deeplab/deeplab_ce-cmap_e100_test1.png")
#let deeplab_ce-cmap_e100_test2 = image(base+"/deeplab/deeplab_ce-cmap_e100_test2.png")
#let deeplab_ce-cmap_e100_train = image(base+"/deeplab/deeplab_ce-cmap_e100_train1.png")

#let deeplab_ce-cmap_test_graph = image(base+"/deeplab/deeplab_ce-cmap_test_graph.png")
#let deeplab_ce-cmap_train_graph = image(base+"/deeplab/deeplab_ce-cmap_train_graph.png")

// =============== U-Net ================
#let unet_ce-cmap_e50_test1 = image(base+"/unet/unet_ce-cmap_e50_test1.png")
#let unet_ce-cmap_e50_test2 = image(base+"/unet/unet_ce-cmap_e50_test2.png")
#let unet_ce-cmap_e50_train = image(base+"/unet/unet_ce-cmap_e50_train1.png")

#let unet_ce-cmap_e100_test1 = image(base+"/unet/unet_ce-cmap_e100_test1.png")
#let unet_ce-cmap_e100_test2 = image(base+"/unet/unet_ce-cmap_e100_test2.png")
#let unet_ce-cmap_e100_train = image(base+"/unet/unet_ce-cmap_e100_train1.png")

#let unet_ce-cmap_test_graph = image(base+"/unet/unet_ce-cmap_test_graph.png")
#let unet_ce-cmap_train_graph = image(base+"/unet/unet_ce-cmap_train_graph.png")

// =============== ViT ================
#let vit_ce-cmap_e50_test1 = image(base+"/vit/vit_ce-cmap_e50_test1.png")
#let vit_ce-cmap_e50_test2 = image(base+"/vit/vit_ce-cmap_e50_test2.png")
#let vit_ce-cmap_e50_train = image(base+"/vit/vit_ce-cmap_e50_train1.png")

#let vit_ce-cmap_e100_test1 = image(base+"/vit/vit_ce-cmap_e100_test1.png")
#let vit_ce-cmap_e100_test2 = image(base+"/vit/vit_ce-cmap_e100_test2.png")
#let vit_ce-cmap_e100_train = image(base+"/vit/vit_ce-cmap_e100_train1.png")

#let vit_ce-cmap_test_graph = image(base+"/vit/vit_ce-cmap_test_graph.png")
#let vit_ce-cmap_train_graph = image(base+"/vit/vit_ce-cmap_train_graph.png")

// =============== Swin ================
#let swin_ce-cmap_e50_test1 = image(base+"/swin/swin_ce-cmap_e50_test1.png")
#let swin_ce-cmap_e50_test2 = image(base+"/swin/swin_ce-cmap_e50_test2.png")
#let swin_ce-cmap_e50_train = image(base+"/swin/swin_ce-cmap_e50_train1.png")

#let swin_ce-cmap_e100_test1 = image(base+"/swin/swin_ce-cmap_e100_test1.png")
#let swin_ce-cmap_e100_test2 = image(base+"/swin/swin_ce-cmap_e100_test2.png")
#let swin_ce-cmap_e100_train = image(base+"/swin/swin_ce-cmap_e100_train1.png")

#let swin_ce-cmap_test_graph = image(base+"/swin/swin_ce-cmap_test_graph.png")
#let swin_ce-cmap_train_graph = image(base+"/swin/swin_ce-cmap_train_graph.png")

== Cross-Entropy + Cold Map

This section will present the results achieved by combining the CE loss with the novel cold map loss. Following the general success of the convolution-based models of understanding the shape and structure of intersections, the results here should show how this great understanding of intersection layout combines with the classification capabilities of the CE loss. Like the other combined training method, here the values of $alpha$ are set to $alpha_"hi" =  0.9$, $alpha_"lo" = 0.5$, and $T_"warm" = 10$. This lower value for $T_"warm"$ is to allow for the cold map loss to take effect sooner in the training process. It also means that the split between CE and the cold map loss will be slightly more gradual than the other combined training method. 

#std-block(breakable: false,
  figure(
    grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: 0mm,
      {deeplab_ce-cmap_e50_test1}, {unet_ce-cmap_e50_test1}, {vit_ce-cmap_e50_test1}, {swin_ce-cmap_e50_test1},
      {deeplab_ce-cmap_e50_test2}, {unet_ce-cmap_e50_test2}, {vit_ce-cmap_e50_test2}, {swin_ce-cmap_e50_test2},
      [#subfigure("(a)") DeepLabv3+], [#subfigure("(b)") U-Net], [#subfigure("(c)") ViT], [#subfigure("(d)") Swin],
    ),
    caption: [Results after the #nth(50) epoch]
  )
)

After just 50 epochs, there are clear trends emerging in the results. It very quickly becomes apparent which sets of models like and dislike the novel loss. The convolution-based models, DeepLabv3+ and U-Net, seem able to take advantage of the cold map loss, as they generally seem to have a better understanding of generating paths that goes out to the edges. Their results are, however, still very unclear. Deeplab is not very confident at marking the path for the roads, and U-Net is not very good at making connected components.

The transformer-based models, ViT and Swin, seem to be struggling with the cold map loss at this early stage. Their results are very sporadic and disconnected. ViT has extremely lumpy and inconsistent results, particularly the lower of the output images, where it generally seems to just try and mark some pixels in the general direction the road is going, without any regard for the connectivity of what it generates. Furthermore, it seems to struggle significantly with even identifying the road, as many of the bumps in its prediction leave the road and there is even a completely detached blob of predicted pixels on top of a house. Swin is showing its own set of problems. Its prediction is way more conservative than ViT, meaning it hardly marks any pixels as being part of the road. 

#std-block(breakable: false,
  figure(
    grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: 0mm,
      {deeplab_ce-cmap_e100_test1}, {unet_ce-cmap_e100_test1}, {vit_ce-cmap_e100_test1}, {swin_ce-cmap_e100_test1},
      {deeplab_ce-cmap_e100_test2}, {unet_ce-cmap_e100_test2}, {vit_ce-cmap_e100_test2}, {swin_ce-cmap_e100_test2},
      [#subfigure("(a)") DeepLabv3+], [#subfigure("(b)") U-Net], [#subfigure("(c)") ViT], [#subfigure("(d)") Swin],
    ),
    caption: [Results after the #nth(100) epoch]
  )
)

This hardly improves for the transformer-based models after 100 epochs. ViT is still very disconnected and lumpy, hardly generating comprehensive results, particularly in the lower image. The random blob on the house is still present, hinting at the model hardly being able to taught from the cold map loss. Swin is still very conservative, still not confident in marking road pixels, but also having ballooned the outputs from the #nth(50) epoch results. It is now marking a lot more pixels as being part of the road, but this largely seems to be done randomly. Generally for these two models, the cold map loss hardly seems to teach them anything about the road structure, as the results are only marginally better than the CE loss alone in that the results are not just thin lines contained to specific internal patches.

Generally for all models, however, seem to be the fact that the mistakes they make earlier in training process are kept as the training progresses and shifts more to rely on the cold map loss. Deeplab shows this by not improving the right turn in the top image; it still consists of a really thin line that seems low in confidence. In the lower image, it also surrounds the left turn pixels with wrongly classified pixels, which started out as a thin underline to the prediction, but is now very dominant with some wrongly classified pixels appearing on top of the path. U-Net is also showing some these persistent problems. The odd gap in the left turn in the top image is still present, despite the fact that the gap is smaller.

=== Training and Validation Graphs

This poor behaviour is also reflected in the training and validation graphs. First, however, it has to be noted that the increase in loss across all models, stem from the fact that the cold map loss values are fairly high. As shown in @fig:cmap_loss_comp, the values for paths semi-close to the ground truth cold map are fairly high, nearing a value of 3. Thus, as the value of $alpha$ shifts towards the cold map loss, the overall will appear to increase more dramatically. Their general shape is still valid for interpretation.

#std-block(breakable: false,
  figure(
    grid(
      columns: (1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: (0mm, 1mm, 0mm, 1mm),
      {deeplab_ce-cmap_train_graph}, 
      {deeplab_ce-cmap_test_graph}, 
      grid.cell([#subfigure("(a)") DeepLabv3+], colspan: 2),
      {unet_ce-cmap_train_graph}, 
      {unet_ce-cmap_test_graph}, 
      grid.cell([#subfigure("(b)") U-Net], colspan: 2),
      {vit_ce-cmap_train_graph}, 
      {vit_ce-cmap_test_graph}, 
      grid.cell([#subfigure("(c)") ViT], colspan: 2),
      {swin_ce-cmap_train_graph},
      {swin_ce-cmap_test_graph},
      grid.cell([#subfigure("(d)") Swin], colspan: 2),
    ),
    caption: [Results after the #nth(100) epoch]
  )
)

Once again, the two sets of models show very different sets of graphs. The convolution-based models do show the continuing trend of a plateauing loss, even as the restarts from the scheduler are applied. DeepLab sees a clear drop in loss after the restarts, and continue to do so for the duration of the training, but U-Net peculiarly only experiences a drop in loss after the first restart. After that, it seems to plateau. This is not present in it accuracy graph, where the dips in accuracy are much more pronounced. The accuracy for both do, however, seem to be on a downward trend, with the plateau being reached at a lower value than before the restarts. DeepLab even shows the training accuracy quickly dropping near the end of training. 

The transformer-based models are much more affected by the restarts, but does swiftly plateau again. When the restarts happen, both models show a decrease in validation loss and an increase in training loss. This is the same behaviour as when they were trained on CE loss alone. An interesting difference between the two models is that ViT seem to perform significantly worse on the validation set, where Swin achieves almost the same accuracy on both sets, even having both training and validation accuracy decrease when the restarts happen. 

=== Test on Training Set

Looking at the results of passing through images from the training set yields some interesting results, highlighting some rather undesired artifacts introduced by the cold map loss. This is particularly apparent with the DeepLab results. After 100 epochs, the model seems to pad the output with a lot of pixels of a seemingly random nature. The other models to not exhibit this behaviour, not even the other convolution-based model, U-Net. The transformer-based models simply generate very thick outputs that are accurately classified.

#std-block(breakable: false,
  figure(
    grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: 0mm,
      {deeplab_ce-cmap_e50_train}, {unet_ce-cmap_e50_train}, {vit_ce-cmap_e50_train}, {swin_ce-cmap_e50_train},
      [#subfigure("(a)") DeepLabv3+], [#subfigure("(b)") U-Net], [#subfigure("(c)") ViT], [#subfigure("(d)") Swin],
    ),
    caption: [Results after the #nth(50) epoch]
  )
)

#std-block(breakable: false,
  figure(
    grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: 0mm,
      {deeplab_ce-cmap_e100_train}, {unet_ce-cmap_e100_train}, {vit_ce-cmap_e100_train}, {swin_ce-cmap_e100_train},
      [#subfigure("(a)") DeepLabv3+], [#subfigure("(b)") U-Net], [#subfigure("(c)") ViT], [#subfigure("(d)") Swin],
    ),
    caption: [Results after the #nth(100) epoch]
  )
)
