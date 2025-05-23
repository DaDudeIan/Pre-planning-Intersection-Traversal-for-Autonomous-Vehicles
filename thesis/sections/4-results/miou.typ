#import "../../lib/mod.typ": *

== mIoU #checked

Finally, this section will present all the mIoU values for the models shown hitherto. This will largely serve to answer #RQ(1). @tab:all_miou is a comparison of the per-class IoU and mean IoU for the four models. The table is sorted in descending order of mIoU, with the best performing model at the top. This is to show which model performs the best, and to gain an overview understanding of the models' performance, especially in comparison to each other.

Concretely, the table reveals that the best performing model is the DeepLabV3+ with the CE loss function, trained for 300 epochs. This model achieved an mIoU of 0.45, which is significantly higher than the other models. Generally, the top is dominated by the DeepLabV3+ model, with the U-Net model performing slightly worse and even better in a few cases. The top 5 are all DeepLabV3+ models, largely consisting of the CE loss, with the appearance of both the combined loss at the third and fourth position. While not impressive in performance, this does show that the loss functions presented, are able to perform at the level of the well-established CE loss function. This is particularly true for the novel cold map loss. Unhappily, however, both of these are from the 50th epoch, where their influence is somewhat small. It is there, but seeing as how the 100th epoch of the same loss combinations are further down the list, it shows that the models are not able to learn from the loss functions when they are equally weighted.

The entire top half of the table is occupied by the convolution-based models, with a significant drop in mIoU to the lower half, consisting only of the transformer-based models. The best performing transformer model is the Swin model, which is at the 15th position with an mIoU of 0.34. This is a significant drop from the top models, and it shows that the transformer models are not able to perform nearly as well as the convolution-based models. This is likely due to the fact that transformers in general require much more data than is provided in the dataset created in this project. Their visual results, however, do show some promise, as they are capable of learning the general structure of the roads and intersections, but need a lot more data to refine their outputs.

In summary, the novel idea of combining the well-established CE loss function with two topology-based loss functions did not yield improved results. None of the models exceed an mIoU of 0.45, indicating that the models are not able to generalize well to the task at hand. Visually, however, many of the models are able to produce outputs that are somewhat usable. Their main downfall comes from not being able to draw fully connected paths and introducing seemingly random artifacts in the generated outputs. The next, penultimate, chapter will discuss the work produced in this thesis project, and the results presented in this section. 

#let tab = [
  #figure(
    {
      tablec(
        columns: 10,
        alignment: (x, y) => (
          right, left, center, center, center, center, center, center, center, center
        ).at(x),
        header: table.header(
          [\#], [Model], [Loss], [Epoch],
          [Class 0], [Class 1], [Class 2], [Class 3], [Class 4], [mIoU $arrow.t$]
        ),

        [1],  [DeepLabV3+], [CE],        [300], [0.9774], [0.3540], [0.3302], [0.3388], [0.2480], [0.4497],
        [2],  [DeepLabV3+], [CE],        [100], [0.9765], [0.3506], [0.3188], [0.3432], [0.2214], [0.4421],
        [3],  [DeepLabV3+], [CE+CMap],   [50],  [0.9701], [0.3214], [0.3078], [0.3216], [0.2143], [0.4271],
        [4],  [DeepLabV3+], [CE+Cont],   [50],  [0.9735], [0.3053], [0.3171], [0.3009], [0.2216], [0.4237],
        [5],  [DeepLabV3+], [CE],        [10],  [0.9695], [0.3344], [0.3253], [0.2983], [0.1772], [0.4209],
        [6],  [U-Net],      [CE+CMap],   [50],  [0.9728], [0.3327], [0.2892], [0.3156], [0.1875], [0.4196],
        [7],  [U-Net],      [CE],        [10],  [0.9704], [0.2952], [0.2858], [0.3276], [0.1735], [0.4105],
        [8],  [DeepLabV3+], [CE+CMap],   [100], [0.9637], [0.2919], [0.2532], [0.3107], [0.2173], [0.4074],
        [9],  [DeepLabV3+], [CE+Cont],   [100], [0.9711], [0.2787], [0.3008], [0.2922], [0.1846], [0.4055],
        [10], [U-Net],      [CE],        [100], [0.9748], [0.3016], [0.2799], [0.2983], [0.1721], [0.4053],
        [11], [U-Net],      [CE+CMap],   [100], [0.9650], [0.2984], [0.2593], [0.2663], [0.1762], [0.3930],
        [12], [U-Net],      [CE+Cont],   [50],  [0.9722], [0.2815], [0.2744], [0.3003], [0.1353], [0.3928],
        [13], [U-Net],      [CE+Cont],   [100], [0.9679], [0.2719], [0.2567], [0.2841], [0.1364], [0.3834],
        [14], [U-Net],      [CE],        [300], [0.9748], [0.2663], [0.2555], [0.2692], [0.1230], [0.3778],
        [15], [Swin],       [CE],        [300], [0.9485], [0.2135], [0.2065], [0.1953], [0.1271], [0.3382],
        [16], [Swin],       [CE],        [100], [0.9490], [0.2009], [0.1895], [0.1681], [0.1294], [0.3274],
        [17], [Swin],       [CE+Cont],   [100], [0.9385], [0.1807], [0.1929], [0.1767], [0.1340], [0.3246],
        [18], [Swin],       [CE+Cont],   [50],  [0.9428], [0.1691], [0.1694], [0.1785], [0.1435], [0.3207],
        [19], [ViT],        [CE],        [100], [0.9453], [0.1664], [0.1523], [0.2065], [0.1211], [0.3183],
        [20], [Swin],       [CE+CMap],   [50],  [0.9323], [0.1824], [0.1533], [0.1579], [0.1348], [0.3122],
        [21], [ViT],        [CE],        [300], [0.9461], [0.1549], [0.1483], [0.1982], [0.1116], [0.3118],
        [22], [ViT],        [CE+Cont],   [100], [0.9281], [0.1617], [0.1592], [0.1805], [0.1268], [0.3113],
        [23], [Swin],       [CE+CMap],   [100], [0.9194], [0.1757], [0.1413], [0.1796], [0.1386], [0.3109],
        [24], [Swin],       [CE],        [10],  [0.9348], [0.1556], [0.1663], [0.1494], [0.1417], [0.3095],
        [25], [ViT],        [CE+CMap],   [100], [0.9185], [0.1442], [0.1502], [0.2162], [0.1016], [0.3061],
        [26], [ViT],        [CE+Cont],   [50],  [0.9295], [0.1553], [0.1593], [0.1775], [0.1015], [0.3046],
        [27], [ViT],        [CE+CMap],   [50],  [0.9218], [0.1472], [0.1361], [0.2001], [0.0890], [0.2988],
        [28], [ViT],        [CE],        [10],  [0.9182], [0.1472], [0.1204], [0.1476], [0.0509], [0.2769],
      )
    },
    caption: [A comparison of the per-class IoU and mean IoU for the four models. Sorted in a descending order of mIoU.]
  )<tab:all_miou>
]

#tab
