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

To create a baseline with which the topology-based loss functions can be compared, the first results to be presented are those from the models trained purely with the cross-entropy loss. This serves as a baseline since CE is a well-documented, well-tested loss function that will help highlight whether or not the topology-based loss function are necessary. Results from different stages of training are shown in #subfigure("Figure 27-29"), with outputs being generated from the models after the #nth(10), #nth(100), and #nth(300) epochs, respectively. 

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
    caption: [Results after the #nth(10) epoch.]
  )
)
The models achieve varying performance after just 10 epochs of training. DeepLab shows strong performance, with it clearly identifying the main structures in the images; it seems to already understand the rules of the road implied by the training data, in that it guides towards staying in the right-hand lane after performing a turn, using the correct turning lane in the top figure, and generally following the path encapsulated by the road markings. It does, however, struggle to mark the exact boundaries of the road, particularly the straight ahead path. Furthermore, it seems to understand the purpose of the `layered` class, in that the right lane, the one shared by going straight ahead and right, is coloured as such, while the left turning pixels are their own colour as it parts ways. U-Net is also showing promising signs of learning the structure of the road, but contains way more gaps and blotches compared it convolutional counterpart. The transformer-based models are not performing well at this stage. Both appear to be grasping the general structure of the road and assigning the correct labels to the road, but the labels are very incomplete and severely lack structure.

#subfigure("Figure 28-29") shows a fairly big leap in training epochs. At this stage, the models have seen the training data for 100 and 300 epochs. The models have evolved both in terms of strengths and weaknesses. At 100 epochs, DeepLab is still the strongest model, but it has become severely uncertain of how to label a path to the end of the road. Its understanding over the structure of the intersection itself is fairly sharp, still showing clear signs of understanding the rules of the road. U-Net has improved close to the level of DeepLab, but still lacks the same level of detail. It is, however, much more certain about the path to the end of the road, as all images shows it identifying the pixels leading out of the satellite image. The transformer-based models, however, have hardly improved. ViT appears to able to create a straight path through the test intersection, but struggles with identifying the road boundaries and creating connected components. Swin suffers from the same downfalls.

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
    caption: [Results after the #nth(100) epoch.]
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
    caption: [Results after the #nth(300) epoch.]
  )
)

At 300 epochs, the models have deteriorated a significant amount. DeepLab is creating more unconnected blotches and the paths it creates are very thin. A thin path is not necessarily a bad thing, but it clearly shows that it is extremely close to breaking the generated paths into more than one component. U-Net shows this even more, where the paths generated are very spread out along the road with many holes in between. The transformer-based models have also not improved further. Both models appear to suffer from the fact that they split the image into patches, with the border of these likely not training well on the data. This is particularly evident in the case of ViT, where the lower image of #subfigure("Figure 29c") shows an odd artefact in the middle of the right-hand side road. It is very jagged and does not fit the road at all. There also appears to be a chunk missing in the labels near the center, hinting at the model learning to not output anything for those specific pixels. Swin is also showing this, but to a lesser extent, likely thanks to its shifting window approach. Its results are still very poor, however. The following graphs will shed some light on why this might be happening. 

=== Training and Validation Graphs


#std-block(breakable: false)[
  #figure(
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
      grid.cell([#subfigure("(d)") Swin], colspan: 2)
    ),
    caption: [Training and Validation graphs for the models trained with cross-entropy loss.],
  ) <fig:ce_graphs>
]

The training and validation graphs for the models trained with cross-entropy loss are shown in @fig:ce_graphs. The saw-tooth like pattern in the graphs is a result of the models being trained using the aforementioned cosine annealing learning rate scheduler. This spikes the learning rate of models at regular intervals. Most commonly, this would results in upward spikes in the graph indicating an increase in loss and downward spikes indicating a decrease in accuracy. The opposite behaviour is apparent in these graphs. 

When a restart occurs from the scheduler, the models significantly improve their performance before sharply rising again. This is very much the opposite of the desired behaviour. However, this is also the expected behaviour of using a cosine annealing scheduler. Typically, this scheduler is employed to help a model escape a poor local minima. This appears to be the case here, but instead of then improving in performance, the models seem to fall into an even worse minima. This appears to be the case for all models, but the strength of the effect varies. The convolutional-based models do not seem to escape a minima that much, as the leap towards a lower loss is not that big and it almost immediately goes beyond what it was before. For the transformer-based models, the effect of escaping the local minima is much more pronounced. The downwards spike jump much farther down and need more time to go back up to the poor loss value. The opposite is here true for the accuracy graphs. 

The restart causes a significant drop in accuracy, but the models quickly recover to the performance they had before the restart. In the case of the convolutional-based models, this is not the case, as the accuracy drops are far shallower. This difference in the saw-tooth pattern is an interesting note between the two types of models. The transformer-based models appear to be more sensitive to the learning rate changes than the convolutional-based models. This is likely due to the fact that the transformer-based models are more complex and have more parameters to learn, meaning they are more sensitive to changes in the learning rate.

The accuracy graphs show the expected behaviour to a very slight extent; the accuracy decreases slightly when the learning rate is increased by the scheduler. All training graphs show very clear signs of overfitting very early in the training. After just a few epochs the loss starts to increase on the validation set, while the training loss continues to decrease. This is a very clear sign of overfitting, as is evident from the outputs of the models when an image from the training dataset is passed through. 

=== Test on Training Set

The input image to the models in this section comes from the training set. Again, the figures shows the outputs of the models after the #nth(10), #nth(100), and #nth(300) epochs. The results are shown in #subfigure("Figure 31-33"). The models have been trained on this data, so it is expected that they perform well on it.

Already at epoch 10, the models are starting to show signs of overfitting. DeepLab and U-Net generate nearly flawless outputs, while the transformer-based models are still struggling. Reaching the #nth(300) epoch shows that even the transformer-based models have learned the training data too well, still to a lesser extent than the convolutional models. A trend of creating thinner and thinner lines in the convolutional models is apparent, showing just how much they have overfitted. 

These illustrations goes well with the observations made in the training and validation graphs. The models are clearly overfitting, as the outputs are very close to the training data. This progressively thinning of the output lines is a clear sign of overfitting, due to the fact that the models are learning to only output the pixels that are occupied by the road with little to no margin for error.

#std-block(breakable: false,
  figure(
    grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      column-gutter: 0mm,
      row-gutter: 0mm,
      {deeplab_ce_e10_train}, {unet_ce_e10_train}, {vit_ce_e10_train}, {swin_ce_e10_train},
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
      {deeplab_ce_e100_train}, {unet_ce_e100_train}, {vit_ce_e100_train}, {swin_ce_e100_train},
      [#subfigure("(a)") DeepLabv3+], [#subfigure("(b)") U-Net], [#subfigure("(c)") ViT], [#subfigure("(d)") Swin],
    ),
    caption: [Results after the #nth(100) epoch.]
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
    caption: [Results after the #nth(300) epoch.]
  )
)


Generally, the results shown in #subfigure("Figure 27-29") shows promise in the task at hand. The models do fairly well when identifying the rules of the roads, but struggles to create a connected path to the end of the road. The transformer-based models are not performing well at all, but they still show signs of learning the general structure of the road, meaning they are likely able to learn the task with more training time and data. The training and validation graphs show that the models are overfitting very early in the training, which is a clear sign of the models not being able to generalize well. 
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

        [DeepLabV3+], [10],  [0.9695], [0.3344], [0.3253], [0.2983], [0.1772], [0.4209],
        [DeepLabV3+], [100], [0.9765], [0.3506], [0.3188], [0.3432], [0.2214], [0.4421],
        [DeepLabV3+], [300], [0.9774], [0.3540], [0.3302], [0.3388], [0.2480], [0.4497],

        [U-Net], [10],      [0.9704], [0.2952], [0.2858], [0.3276], [0.1735], [0.4105],
        [U-Net], [100],     [0.9748], [0.3016], [0.2799], [0.2983], [0.1721], [0.4053],
        [U-Net], [300],     [0.9748], [0.2663], [0.2555], [0.2692], [0.1230], [0.3778],

        [ViT], [10],        [0.9182], [0.1472], [0.1204], [0.1476], [0.0509], [0.2769],
        [ViT], [100],       [0.9453], [0.1664], [0.1523], [0.2065], [0.1211], [0.3183],
        [ViT], [300],       [0.9461], [0.1549], [0.1483], [0.1982], [0.1116], [0.3118],

        [Swin], [10],       [0.9348], [0.1556], [0.1663], [0.1494], [0.1417], [0.3095],
        [Swin], [100],      [0.9490], [0.2009], [0.1895], [0.1681], [0.1294], [0.3274],
        [Swin], [300],      [0.9485], [0.2135], [0.2065], [0.1953], [0.1271], [0.3382],

        []
      )
    },
    caption: [Per-class IoU and mean IoU for the four models trained with plain CE loss at 10, 100, and 300 epochs.]
  )<tab:ce_miou>
]

Supporting this observation is @tab:ce_miou, showcasing that the models are not performing well at all. The mIoU is very low, with the best performing model, DeepLab, achieving a mIoU of 0.45 after 300 epochs. The other models are not far behind with U-Net achieving a mIoU of 0.41, following a drop is ViT achieving a mIoU of 0.32, and Swin achieving a mIoU of 0.34. The per-class are also very poor, with DeepLab achieving a loss of around 0.34 for each of the directions through an intersection, with the layered class being the worst performing class. This also holds true for the other models, just with much lower values. This does seem to mark a positive trend, where the models understand that an even spread of the classes is expected. To help with the structure of the output, the next section will present the results of the models trained with CE and the continuity loss.
#tab
