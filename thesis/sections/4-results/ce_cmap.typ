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

== Cross-Entropy + Cold Map #checked

This section will present the results achieved by combining the CE loss with the novel cold map loss. Following the general success of the convolution-based models of understanding the shape and structure of intersections, the results here should show how this great understanding of intersection layout combines with the classification capabilities of the CE loss. Like the other combined training method, here the values of $alpha$ are set to $alpha_"hi" =  0.9$, $alpha_"lo" = 0.5$, and $T_"warm" = 10$. This lower value for $T_"warm"$ is to allow for the cold map loss to take effect sooner in the training process. It also means that the split between CE and the cold map loss will be slightly more gradual than the other combined training method. 

#std-block(breakable: false,
  figure(
    grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: 0mm,
      {deeplab_ce-cmap_e50_test1}, {unet_ce-cmap_e50_test1}, {vit_ce-cmap_e50_test1}, {swin_ce-cmap_e50_test1},
      {deeplab_ce-cmap_e50_test2}, {unet_ce-cmap_e50_test2}, {vit_ce-cmap_e50_test2}, {swin_ce-cmap_e50_test2},
      [#subfigure("(a)") DeepLabV3+], [#subfigure("(b)") U-Net], [#subfigure("(c)") ViT], [#subfigure("(d)") Swin],
    ),
    caption: [Results after the #nth(50) epoch. These results are very much like those of the CE standalone, as the cold map loss has hardly had the change to take effect. The transformer models seem particularly dislike the cold map loss, as their outputs have degraded from the baseline.]
  )
)

After just 50 epochs, there are clear trends emerging in the results. It very quickly becomes apparent which sets of models like and dislike the novel loss. The convolution-based models, DeepLabV3+ and U-Net, seem able to take advantage of the cold map loss, as they generally seem to have a better understanding of generating paths that goes out to the edges. Their results are, however, still very unclear. DeepLab is not very confident at marking the path for the roads, and U-Net is not very good at making connected components.

The transformer-based models, ViT and Swin, seem to be struggling with the cold map loss at this early stage. Their results are very sporadic and disconnected. ViT has extremely lumpy and inconsistent results, particularly in the lower of the output images, where it generally seems to just try and mark some pixels in the general direction the road is going, without any regard for the connectivity of what it generates. Furthermore, it seems to struggle significantly with even identifying the road, as many of the bumps in its prediction leave the road and there is even a completely detached blob of predicted pixels on top of a house. Swin is showing its own set of problems. Its prediction is way more conservative than ViT, meaning it hardly marks any pixels as being part of the road. 

#std-block(breakable: false,
  figure(
    grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: 0mm,
      {deeplab_ce-cmap_e100_test1}, {unet_ce-cmap_e100_test1}, {vit_ce-cmap_e100_test1}, {swin_ce-cmap_e100_test1},
      {deeplab_ce-cmap_e100_test2}, {unet_ce-cmap_e100_test2}, {vit_ce-cmap_e100_test2}, {swin_ce-cmap_e100_test2},
      [#subfigure("(a)") DeepLabV3+], [#subfigure("(b)") U-Net], [#subfigure("(c)") ViT], [#subfigure("(d)") Swin],
    ),
    caption: [Results after the #nth(100) epoch. The artifacts from the cold map standalone loss become apparent in DeepLab's outputs, and the transformer models are more likely to generate seemingly random blotches.]
  )
)

This hardly improves for the transformer-based models after 100 epochs. ViT is still very disconnected and lumpy, hardly generating comprehensive results, particularly in the lower image. The random blob on the house is still present, hinting at the model hardly being able to be taught from the cold map loss. Swin is still very conservative, still not confident in marking road pixels, but also having ballooned the outputs from the #nth(50) epoch results. It is now marking a lot more pixels as being part of the road, but this largely seems to be done randomly. Generally for these two models, the cold map loss hardly seems to teach them anything about the road structure, as the results are only marginally better than the CE loss alone in that the results are not just thin lines contained to specific internal patches.

Generally for all models, however, seem to be the fact that the mistakes they make earlier in training process are kept as the training progresses and shifts more to rely on the cold map loss. DeepLab shows this by not improving the right turn in the top image; it still consists of a really thin line that seems low in confidence. In the lower image, it also surrounds the left turn pixels with wrongly classified pixels, which started out as a thin underline to the prediction, but is now very dominant with some wrongly classified pixels appearing on top of the path. U-Net is also showing some these persistent problems. The odd gap in the left turn in the top image is still present, despite the fact that the gap is smaller.

=== Training and Validation Graphs #checked


#std-block(breakable: false,
  figure(
    grid(
      columns: (1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: (0mm, 1mm, 0mm, 1mm),
      {deeplab_ce-cmap_train_graph}, 
      {deeplab_ce-cmap_test_graph}, 
      grid.cell([#subfigure("(a)") DeepLabV3+], colspan: 2),
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
    caption: [Training and Validation graphs for the models trained with CE and cold map losses.]
  )
)

This poor behaviour is also reflected in the training and validation graphs. First, however, it has to be noted that the increase in loss across all models, stem from the fact that the cold map loss values are fairly high. As shown in @fig:cmap_loss_comp, the values for paths semi-close to the ground truth cold map are fairly high, nearing a value of 3. Thus, as the value of $alpha$ shifts towards the cold map loss, the overall will appear to increase more dramatically. Their general shape is still valid for interpretation.

Once again, the two sets of models show very different sets of graphs. The convolution-based models do show the continuing trend of a plateauing loss, even as the restarts from the scheduler are applied. DeepLab sees a clear drop in loss after the restarts, and continue to do so for the duration of the training, but U-Net peculiarly only experiences a drop in loss after the first restart. After that, it seems to plateau. This is not present in its accuracy graph, where the dips in accuracy are much more pronounced. The accuracy for both do, however, seem to be on a downward trend, with the plateau being reached at a lower value than before the restarts. DeepLab even shows the training accuracy quickly dropping near the end of training. 

The transformer-based models are much more affected by the restarts, but swiftly plateaus again. When the restarts happen, both models show a decrease in validation loss and an increase in training loss. This is the same behaviour as when they were trained on CE loss alone. An interesting difference between the two models is that ViT seem to perform significantly worse on the validation set, where Swin achieves almost the same accuracy on both sets, even having both training and validation accuracy decrease when the restarts happen. 

=== Test on Training Set #checked

Looking at the results of passing through images from the training set yields some interesting results, highlighting some rather undesired artifacts introduced by the cold map loss. This is particularly apparent with the DeepLab results. After 100 epochs, the model seems to pad the output with a lot of pixels of a seemingly random nature. The other models do not exhibit this behaviour, not even the other convolution-based model, U-Net. The transformer-based models simply generate very thick outputs that are accurately classified.


#std-block(breakable: false,
  figure(
    grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: 0mm,
      {deeplab_ce-cmap_e50_train}, {unet_ce-cmap_e50_train}, {vit_ce-cmap_e50_train}, {swin_ce-cmap_e50_train},
      [#subfigure("(a)") DeepLabV3+], [#subfigure("(b)") U-Net], [#subfigure("(c)") ViT], [#subfigure("(d)") Swin],
    ),
    caption: [Results after the #nth(50) epoch. These results are very much like those of the CE standalone, as the cold map loss has hardly had the change to take effect. The transformer models, however, appear more bloated than previous results.]
  )
)

Interestingly, the cold map appear to introduce some level of disparity in the results, as shown in @tab:ce-cmap_miou. Particularly, the three main classes are far less balanced than either @tab:ce_miou or @tab:ce-cont_miou show. In the case of class 2, right-hand turns #ball(color.rgb("#bb3754")), the models seem to choose it way less often than the other classes. This is supported by @fig:ce-cmap_train_results_100#subfigure("a"), where the path going left is surrounded by pixels belonging to the class going right ans straight. These will in turn be used to calculate the IoU for the class going right, meaning they will be lower than they should. This is, however, still something that should be considered with this novel loss, as it is an artifact not present in the other losses.
#std-block(breakable: false)[
  #figure(
    grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: 0mm,
      {deeplab_ce-cmap_e100_train}, {unet_ce-cmap_e100_train}, {vit_ce-cmap_e100_train}, {swin_ce-cmap_e100_train},
      [#subfigure("(a)") DeepLabV3+], [#subfigure("(b)") U-Net], [#subfigure("(c)") ViT], [#subfigure("(d)") Swin],
    ),
    caption: [Results after the #nth(100) epoch. Once again, the artifacts from the cold map standalone loss become apparent in DeepLab's outputs, and the transformer models are more likely to generate seemingly random blotches. U-Net is largely unaffected, but instead of overfitting and making thinner lines, the paths are thicker.]
  ) <fig:ce-cmap_train_results_100>
]


#let tab = [
  #figure(
    {
      tablec(
        columns: 8,
        alignment: (x, y) => (left, center, center, center, center, center, center, center).at(x),
        header: table.header(
          [Model], [Epoch], [Class 0], [Class 1], [Class 2],
          [Class 3], [Class 4], [mIoU $arrow.t$]
        ),

        [DeepLabV3+], [50],  [0.9701], [0.3214], [0.3078], [0.3216], [0.2143], [0.4271],
        [DeepLabV3+], [100], [0.9637], [0.2919], [0.2532], [0.3107], [0.2173], [0.4074],

        [U-Net], [50],       [0.9728], [0.3327], [0.2892], [0.3156], [0.1875], [0.4196],
        [U-Net], [100],      [0.9650], [0.2984], [0.2593], [0.2663], [0.1762], [0.3930],

        [ViT], [50],         [0.9218], [0.1472], [0.1361], [0.2001], [0.0890], [0.2988],
        [ViT], [100],        [0.9185], [0.1442], [0.1502], [0.2162], [0.1016], [0.3061],

        [Swin], [50],        [0.9323], [0.1824], [0.1533], [0.1579], [0.1348], [0.3122],
        [Swin], [100],       [0.9194], [0.1757], [0.1413], [0.1796], [0.1386], [0.3109],

        []
      )
    },
    caption: [Per-class IoU and mean IoU for the same backbones trained with CE + cold map loss at 50 and 100 epochs.]
  )<tab:ce-cmap_miou>
]


The model with the best performance is still DeepLabV3+, but it is not as dominant as it was with the CE loss alone. U-Net is very close behind, with ViT and Swin trailing further behind. DeepLab achieves a slightly higher mIoU compared to the other combined loss, in that it here achieves a mIoU of 0.427, compared to 0.424. This is a negligible difference, especially when the artifacts introduced visually create more obscure results.

#tab