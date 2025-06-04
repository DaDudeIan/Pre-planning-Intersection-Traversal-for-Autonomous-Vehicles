#import "../../lib/mod.typ": *

#let base = "../../figures/img/results"

// =============== Deeplab ================
#let deeplab_ce-topo_e50_test1 = image(base+"/deeplab/deeplab_ce-topo_e50_test1.png")
#let deeplab_ce-topo_e50_test2 = image(base+"/deeplab/deeplab_ce-topo_e50_test2.png")
#let deeplab_ce-topo_e50_train = image(base+"/deeplab/deeplab_ce-topo_e50_train1.png")

#let deeplab_ce-topo_e100_test1 = image(base+"/deeplab/deeplab_ce-topo_e100_test1.png")
#let deeplab_ce-topo_e100_test2 = image(base+"/deeplab/deeplab_ce-topo_e100_test2.png")
#let deeplab_ce-topo_e100_train = image(base+"/deeplab/deeplab_ce-topo_e100_train1.png")

#let deeplab_ce-topo_test_graph = image(base+"/deeplab/deeplab_ce-topo_test_graph.png")
#let deeplab_ce-topo_train_graph = image(base+"/deeplab/deeplab_ce-topo_train_graph.png")

// =============== U-Net ================
#let unet_ce-topo_e50_test1 = image(base+"/unet/unet_ce-topo_e50_test1.png")
#let unet_ce-topo_e50_test2 = image(base+"/unet/unet_ce-topo_e50_test2.png")
#let unet_ce-topo_e50_train = image(base+"/unet/unet_ce-topo_e50_train1.png")

#let unet_ce-topo_e100_test1 = image(base+"/unet/unet_ce-topo_e100_test1.png")
#let unet_ce-topo_e100_test2 = image(base+"/unet/unet_ce-topo_e100_test2.png")
#let unet_ce-topo_e100_train = image(base+"/unet/unet_ce-topo_e100_train1.png")

#let unet_ce-topo_test_graph = image(base+"/unet/unet_ce-topo_test_graph.png")
#let unet_ce-topo_train_graph = image(base+"/unet/unet_ce-topo_train_graph.png")

// =============== ViT ================
#let vit_ce-topo_e50_test1 = image(base+"/vit/vit_ce-topo_e50_test1.png")
#let vit_ce-topo_e50_test2 = image(base+"/vit/vit_ce-topo_e50_test2.png")
#let vit_ce-topo_e50_train = image(base+"/vit/vit_ce-topo_e50_train1.png")

#let vit_ce-topo_e100_test1 = image(base+"/vit/vit_ce-topo_e100_test1.png")
#let vit_ce-topo_e100_test2 = image(base+"/vit/vit_ce-topo_e100_test2.png")
#let vit_ce-topo_e100_train = image(base+"/vit/vit_ce-topo_e100_train1.png")

#let vit_ce-topo_test_graph = image(base+"/vit/vit_ce-topo_test_graph.png")
#let vit_ce-topo_train_graph = image(base+"/vit/vit_ce-topo_train_graph.png")

// =============== Swin ================
#let swin_ce-topo_e50_test1 = image(base+"/swin/swin_ce-topo_e50_test1.png")
#let swin_ce-topo_e50_test2 = image(base+"/swin/swin_ce-topo_e50_test2.png")
#let swin_ce-topo_e50_train = image(base+"/swin/swin_ce-topo_e50_train1.png")

#let swin_ce-topo_e100_test1 = image(base+"/swin/swin_ce-topo_e100_test1.png")
#let swin_ce-topo_e100_test2 = image(base+"/swin/swin_ce-topo_e100_test2.png")
#let swin_ce-topo_e100_train = image(base+"/swin/swin_ce-topo_e100_train1.png")

#let swin_ce-topo_test_graph = image(base+"/swin/swin_ce-topo_test_graph.png")
#let swin_ce-topo_train_graph = image(base+"/swin/swin_ce-topo_train_graph.png")

== Cross-Entropy + Continuity  

Before moving on to the novel cold map approach, the results of using the existing topology-based loss method will be shown. This combination of loss functions, should present better results than that of the standalone CE loss, as it is topology-unaware and the only kind of topological control it has, comes from the fact that the outputs are structured fairly consistently. Thus, the results here should show less of a deterioration of the topology of the outputs, compared to the standalone CE loss. 

As presented in @c4:training-strategy, the training of these combined loss functions was done using a dynamic value for $alpha$ which is a term used to combine two loss function, with their contributions not excessing 1 in order to have stable training. The value of $alpha$ for this setup was with $alpha_"hi" =  0.99$, $alpha_"lo" = 0.5$, and $T_"warm" = 30$. Therefore, the results are only shown from the #nth(50) and #nth(100) epoch checkpoints, as the impact of the continuity loss is negligible until it passes the #nth(30) epoch. 

At the #nth(50) epoch, the results only show marginal improvement, most notably shown in the ViT outputs. This is expected, as the $alpha$ value is slowly shifting in favour of the continuity loss, meaning that the CE loss still holds a vast majority of the contributions to the weight updates. As of this stage, however, it is not looking too promising, as the outputs from the DeepLab models shows various artifacts in the form of little flakes that seem to align themselves with the road markings at the immediate exit of the straight-ahead path #ball(blue). Otherwise, its outputs are fairly consistent as it has already shown in the previous section. The U-Net outputs are also consistent with the previous section, as it still isn't too confident with slightly unclear roads, such as the right-hand turn in the top image #ball(green), and the entry road in the bottom image #ball(yellow). 

#std-block(breakable: false)[
  #figure(
    grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: 0mm,
      {deeplab_ce-topo_e50_test1}, [#unet_ce-topo_e50_test1 #place(center + horizon, dy: -0.5mm, dx: 1.08cm, rect(height: 5mm, width: 1.5cm, stroke: 1pt + green))], {vit_ce-topo_e50_test1}, {swin_ce-topo_e50_test1},
      [#deeplab_ce-topo_e50_test2 #place(center + horizon, dy: -5.5mm, dx: 1.2mm, ellipse(height: 1cm, width: 5mm, stroke: 1pt + blue))], [#unet_ce-topo_e50_test2 #place(center + horizon, dy: 10mm, dx: 0mm, rect(height: 14mm, width: 5mm, stroke: 1pt + yellow))], {vit_ce-topo_e50_test2}, {swin_ce-topo_e50_test2},
      [#subfigure("(a)") DeepLabV3+], [#subfigure("(b)") U-Net], [#subfigure("(c)") ViT], [#subfigure("(d)") Swin],
    ),
    caption: [Results after the #nth(50) epoch. At this stage, the continuity loss is not yet having a large impact on the outputs, as the $alpha$ value is still largely in favour of the CE loss. ]
  ) <fig:res_ce-topo-e50>
]

#std-block(breakable: false,
  figure(
    grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: 0mm,
      [#deeplab_ce-topo_e100_test1], {unet_ce-topo_e100_test1}, {vit_ce-topo_e100_test1}, {swin_ce-topo_e100_test1},
      [#deeplab_ce-topo_e100_test2 #place(center + horizon, dy: -5.5mm, dx: 1.2mm, ellipse(height: 1cm, width: 5mm, stroke: 1pt + red))], {unet_ce-topo_e100_test2}, {vit_ce-topo_e100_test2}, {swin_ce-topo_e100_test2},
      [#subfigure("(a)") DeepLabV3+], [#subfigure("(b)") U-Net], [#subfigure("(c)") ViT], [#subfigure("(d)") Swin],
    ),
    caption: [Results after the #nth(100) epoch. The outputs from the models are generally more connected, with fewer disconnected components. ]
  )
)

As have been pointed out already, the outputs from the ViT model are showing a lot of promise, as the output consists of just two major blobs of output in the top image, while the angled roads of the seconds still need some work. As it also showed with the CE loss, it is not good with using the layered #ball(color.rgb("#fcfea4")) class label, seemingly only applying it to the same internal patches each time. The Swin model is largely the same, as it appears to be better at outputting just one component, while also correctly identifying which class is to be used where.

At the later stages of training, the continuity loss has a large impact on the outputs, as the split between the two losses is 50-50. This is very predominant in all the models' outputs. The DeepLab model's outputs show a significant improvement, as many of the spurious artifacts are no longer present #ball(blue)$arrow$#ball(red). The paths generated shows the desired improvement, while still keeping its understanding of the rules of the road. This is shown by the fact that, as mentioned, the outputs are cleaner and they are still correctly labelling the turns through the intersection, ending on the right side of the road.

This is to a lesser extent true for the U-Net model. While largely consistent of the one desired component, it has become uncertain of how to enter and exit the intersection. The top image shows some good results, but it doesn't seem to be able to finish the right-hand turn to the edge of the image #ball(green), but the output is one component. For the bottom image, however, it is not able to correctly identify the entry pixels #ball(yellow), but it does only consist of two, fairly large components. 

The transformer models appear to have learnt a lot from the continuity loss, as they generally output one component, with some smaller or larger blotches along the road. Particularly for the first image does the output look very good, as it consists of a few large components, that are likely to merge after some more training. Both models also show their weakness in the bottom image, as neither model's outputs are particularly coherent. Generally, the desired effect of introducing an established topology-based loss function has been achieved, mainly in the regard that smaller, disconnected components are largely absent when compared to the pure CE loss scheme.

=== Training and Validation Graphs  

The training and validation graphs for the models trained with CE and continuity losses are shown in @fig:ce-topo_graphs. Once again, the saw-tooth pattern in the graphs are present, still due to the fact that the models are trained with a cosine annealing scheduler. The graphs do, however, appear to be more erratic than the ones from the CE loss. This holds particularly true for the accuracy graphs of the transformer-based models.

The convolutional-based models show the same problematic behaviour in their loss graphs; the validation loss continues to increase significantly while the training loss is slowly plateauing, only being disturbed by the restarts of the scheduler. The accuracy graphs are also showing a similar pattern, but with only small perturbations happening at the restarts, after which they quickly settle again. Interestingly, the graphs for the training and validation accuracy are very similar in shape and reaction to the restarts. This suggests that the models are not benefiting that much from the topology loss. This is the opposite of what can be seen in the transformer models' graphs.

The transformer models, however, show very different behaviour. The training loss graphs show the same saw-tooth pattern as the convolutional models, but the validation loss graphs are much more erratic. First, however, it is worth noting that the loss does not rise to what it was before the restart happens, but rather settles at a lower value. This is promising, as it shows that the model is learning something by escaping from the local minima it might have been stuck in and found another, better one to settle into. Their accuracy graphs are extremely erratic by comparison. 

#std-block(breakable: false)[
  #figure(
    grid(
      columns: (1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: (0mm, 1mm, 0mm, 1mm),
      {deeplab_ce-topo_train_graph}, 
      {deeplab_ce-topo_test_graph}, 
      grid.cell([#subfigure("(a)") DeepLabV3+], colspan: 2),
      {unet_ce-topo_train_graph}, 
      {unet_ce-topo_test_graph}, 
      grid.cell([#subfigure("(b)") U-Net], colspan: 2),
      {vit_ce-topo_train_graph}, 
      {vit_ce-topo_test_graph}, 
      grid.cell([#subfigure("(c)") ViT], colspan: 2),
      {swin_ce-topo_train_graph},
      {swin_ce-topo_test_graph},
      grid.cell([#subfigure("(d)") Swin], colspan: 2),
    ),
    caption: [Training and Validation graphs for the models trained with CE and continuity losses.]
  ) <fig:ce-topo_graphs>
]



The training accuracy nicely follows the same pattern as the loss graphs, but the validation accuracy is all over the place. This hints at the fact that, the models may in reality not work particularly well with the introduced topology-based loss function, as the mean of spiky graphs appears to be fairly consistent, at around 92% for ViT and 93% for Swin, with both having massive fluctuations in both the positive and negative direction. This is very unlike the purely CE trained models, where the accuracy graphs are closely following the same pattern, for both the training and validation accuracy.

=== Test on Training Set  


#std-block(breakable: false,
  figure(
    grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: 0mm,
      {deeplab_ce-topo_e50_train}, {unet_ce-topo_e50_train}, {vit_ce-topo_e50_train}, {swin_ce-topo_e50_train},
      [#subfigure("(a)") DeepLabV3+], [#subfigure("(b)") U-Net], [#subfigure("(c)") ViT], [#subfigure("(d)") Swin],
    ),
    caption: [Results after the #nth(50) epoch. The same overfitting behaviour is present as the CE standalone loss, but the transformer models are generating less accurate paths.]
  )
)

#std-block(breakable: false,
  figure(
    grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: 0mm,
      {deeplab_ce-topo_e100_train}, {unet_ce-topo_e100_train}, {vit_ce-topo_e100_train}, {swin_ce-topo_e100_train},
      [#subfigure("(a)") DeepLabV3+], [#subfigure("(b)") U-Net], [#subfigure("(c)") ViT], [#subfigure("(d)") Swin],
    ),
    caption: [Results after the #nth(100) epoch. The transformer models are still generating outputs that are significantly worse than the convolutional models.]
  )
)

This erratic behaviour is also present in the test set results. Expectedly, the convolutional-based models are very overfitted to the training set, so their outputs are hardly influenced by the topology-based loss. But the transformer models' outputs show why they generate such erratic graphs. For both models, the outputs w.r.t. the left- and right-hand turns are extremely bumpy and does not follow the shape of the road at all. As evident by the graphs, these lumpy outputs change a lot during training, likely going from being bumpy to one side of the road to the other, meaning the models are very slowly learning the task at hand, without showing any major improvements.

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

        [DeepLabV3+], [50],  [0.9735], [0.3053], [0.3171], [0.3009], [0.2216], [0.4237],
        [DeepLabV3+], [100], [0.9711], [0.2787], [0.3008], [0.2922], [0.1846], [0.4055],

        [U-Net], [50],       [0.9722], [0.2815], [0.2744], [0.3003], [0.1353], [0.3928],
        [U-Net], [100],      [0.9679], [0.2719], [0.2567], [0.2841], [0.1364], [0.3834],

        [ViT], [50],         [0.9295], [0.1553], [0.1593], [0.1775], [0.1015], [0.3046],
        [ViT], [100],        [0.9281], [0.1617], [0.1592], [0.1805], [0.1268], [0.3113],

        [Swin], [50],        [0.9428], [0.1691], [0.1694], [0.1785], [0.1435], [0.3207],
        [Swin], [100],       [0.9385], [0.1807], [0.1929], [0.1767], [0.1340], [0.3246],

        []
      )
    },
    caption: [Per-class IoU and mean IoU for the four models trained with combined cross-entropy + continuity loss at 50 and 100 epochs.]
  )<tab:ce-cont_miou>
]

#tab

@tab:ce-cont_miou also shows some disappointing results w.r.t the per-class IoU and mIoU. Compared to @tab:ce_miou, the results have seemingly only gotten worse. The DeepLab model's outputs are still the best performing, but the mIoU has dropped from 0.45 to 0.42, which further dropped as training progressed to 0.41. It has seemingly become less confident in the three main classes, as the IoU per classes 1-3 has dropped to below 0.3. A similar drop is seen in all the other models, with the U-Net model dropping from 0.41 to 0.39, and the transformer models dropping from 0.32 to 0.30 for the ViT and from 0.34 to 0.32 for the Swin model.  

In summary, the results of the models trained with the CE and continuity loss function show that the topology-based loss function does have a positive impact on the outputs, visually. The main success comes from the fact that small artifacts are largely absent in the outputs, with the models having learnt to generate few large, connected components, often only generating the desired singular component. Although their results are cleaner, in the long run, the convolutional-based models show little improvement, as they only removed minor artifacts that could be removed with some post-processing. The transformer models, however, seem to be largely affected by the addition of the continuity loss, in that their training and validation graphs are very erratic hinting at an unstable training process. Contrary to the visual improvement, the mIoU shown in the table, indicates that the models are not really improving with the addition of the continuity loss. With these results, the novel cold map loss results will now be shown, first as a standalone method, and then in combination with the CE loss.