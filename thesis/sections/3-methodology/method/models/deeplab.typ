#import "../../../../lib/mod.typ": *
=== DeepLabV3+ <c4:deeplab>

#let convolution_colour = rgb("7EA6E0")
#let Down_colour = rgb("FF3333")
#let aspp_colour = rgb("66CC00")
#let imgpooling_colour = rgb("FFB366")
#let encoder_colour = rgb("B3FF66")
#let decoder_colour = rgb("FFCE9F")
#let low_level_colour = rgb("9AC7BF")

// Hello #ball(convolution_colour) #ball(Down_colour) #ball(aspp_colour) #ball(imgpooling_colour) #ball(encoder_colour) #ball(decoder_colour)

The DeepLab family of models is a series of models designed for semantic segmentation tasks by Google. The original DeepLab model was introduced in 2016 and has since evolved into several versions, with DeepLabV3+ being the latest iteration. 

DeepLab, contextually DeepLabV1, was the first model of the family to be released in 2016 @deeplabv1. It introduced several innovations, chief amongst which was the idea of using atrous convolution, also known as dilated convolution. Shown in @fig:atrous_conv, atrous convolution is a method where holes are introduced in the convolutional kernels. This allows each pixel in the resulting feature map to capture a wider context, without having to using massive kernels. Formally, this technique allows for the extraction of multi-scale features without losing resolution, making it particularly effective for semantic segmentation tasks. Furthermore, the model, and its successor, used a fully connected conditional random field (CRF). This is a post-processing step used to refine the pixel-level labellings produced by the network. It operates by defining an energy function that includes costs for assigning labels to individual pixels, called unary potentials, and costs based on the label assignments of all possible pairs of pixels, called pairwise potentials. The successor, DeepLabV2, also used this post-processing step, but not in later versions. 

DeepLabV2, utilized the concept of #acr("ASPP"), which employs multiple parallel atrous convolutions with different rates to capture features at various scales. The "pyramid" part of its name comes from the fact that it uses multiple atrous convolutions with different rates, effectively creating a pyramid of features at different scales. The introduction of this technique can be compared to the popularization of skip connection's usage in U-Net, garnering over 25,000 citations.

#let fig1 = { image("../../../../figures/img/models/deeplab/atrous.png") }
#let my_brace = [#v(-12pt)$underbrace(#box(width: 100%))$]
#let a_block = 1fr
#let a_space = 0.07fr

#std-block(breakable: false)[
  #figure( grid( columns: (a_block, a_space, a_block, a_space, a_block, a_space, a_block), column-gutter: 0mm,
    grid.cell(fig1, colspan: 7),
    my_brace, [], my_brace, [], my_brace, [], my_brace,
    [3x3 Convolution, \ Dilation: 1.], [], [3x3 Convolution, \ Dilation: 6.], [], [3x3 Convolution, \ Dilation: 12.], [], [3x3 Convolution, \ Dilation: 18.]

  ),
  caption: [Atrous Convolution.]
) <fig:atrous_conv>
]
// Fig. 4: Atrous Spatial Pyramid Pooling (ASPP). To classify the center pixel (orange), ASPP exploits multi-scale features by employing multiple parallel filters with different rates. 

DeepLabV3 was released in 2017 @deeplabv3, introducing an enhanced ASPP module, image pooling, and batch normalization. The ASPP module was improved by adding image pooling, which allows the model to capture global context information. This is done by applying a global average pooling operation to the feature map. Finally, DeepLabV3+ was released in 2018 @deeplabv3plus, which turned the DeepLab architecture into an encoder-decoder structure. This was done by adding a decoder module to the DeepLabV3 architecture. So, the encoder part #ball(encoder_colour) of the models is the same as the DeepLabV3 model, which consists of a backbone network, in this project MobileNetV3 @mobilenetv3, and the ASPP module. The top half of @fig:deeplabv3 shows the encoder part of the model. Five features maps are concatenated in the encoder. First, a simple 1x1 convolution #ball(convolution_colour) is applied. Second, atrous convolution is used to create 3 different feature maps #ball(aspp_colour), each with a different dilation rate. Finally, an image pooling #ball(imgpooling_colour) feature map is concatenated to capture global context. 

In the decoder part #ball(decoder_colour), the model uses a low-level feature map #ball(low_level_colour) from the encoder. This feature map is concatenated with the output of the ASPP module after being passed through another 1x1 convolution #ball(Down_colour). A final segmentation head is then applied to the concatenated feature map before being upscaled to the original image size. The segmentation head consists of a 3x3 convolutional layer followed by a bilinear upsampling layer. This allows the model to produce a segmentation map that is the same size as the input image. 

This family of models achieves high results on various datasets. From the outset, DeepLabV1 achieved a mean intersection over union (mIoU) of 71.6% on the PASCAL VOC 2012 dataset using VGG-16 as the backbone of the network. DeepLabV2 improved this to 79.7% mIoU using ResNet-101 as the backbone on the same dataset. DeepLabV3 further improved this to 82.7% mIoU using ResNet-101 as the backbone on the PASCAL VOC 2012 dataset. Finally, DeepLabV3+ achieved 89.0% mIoU using Xception as the backbone on the same dataset, and 82.1% mIoU on the Cityscapes dataset. This shows that the DeepLab family of models is capable of achieving high results on various datasets. 

#let fig1 = { image("../../../../figures/img/models/deeplab/architecture.png") }

#std-block(breakable: false)[
  #figure( fig1,
  caption: [DeepLabV3+ Architecture.]
) <fig:deeplabv3>
]

The implementation of this model follows the network shown in @fig:deeplabv3. The backbone, as mentioned, is the MobileNetV3 network. This network was chosen as it is a lightweight network, capable of running on the machine used in this project and for a more simplistic implementation. Other backbones are available and have active communities. The original paper was proposed with the ResNet-101 and Xception backbones, considered to be classic CNNs. Larger networks like EfficientNet-L2 and ConvNeXt are also viable options for more complex tasks. The base version of ConvNeXt was tested, namely ConvNeXt-base, but simply crashed when initialized. There's even work being done to integrate the DeepLab architecture with transformer-based backbones, such as ViT and Swin Transformer.

In summary, the two convolution-based models chosen for this project are the U-Net and DeepLabV3+ models. The U-Net model is a fully convolutional network that uses skip connections to retain spatial information, while the DeepLabV3+ model uses atrous convolution and ASPP to capture multi-scale features. Both models have been shown to achieve high results on various datasets, making them suitable for the task at hand. The next sections will present the transformer-based models chosen for this project.