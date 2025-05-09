#import "../../../../lib/mod.typ": *
=== U-Net #checked <c4:unet>

#let DoubleConv_colour = rgb("7EA6E0")
#let Down_colour = rgb("FF3333")
#let Up_colour = rgb("66CC00")
#let OutConv_colour = rgb("FFB366")

// Hello #ball(DoubleConv_colour) #ball(Down_colour) #ball(Up_colour) #ball(OutConv_colour)

// #let b1 = std-block(breakable: false)[
//   #box(
//     fill: DoubleConv_colour,
//     outset: 1mm,
//     inset: 0em,
//     radius: 3pt,
//   )[#text(white, size: 12pt, font: "JetBrainsMono NFM")[DoubleConv]] \
//   #listing(line-numbering: none, [
//     ```python
// class DoubleConv(nn.Module):
// def __init__(self, in_c, out_c):
//   super(DoubleConv, self).__init__()
//   self.double_conv = nn.Sequential(
//     nn.Conv2d(...),
//     nn.BatchNorm2d(...),
//     nn.ReLU(...),
//     nn.Conv2d(...),
//     nn.BatchNorm2d(...),
//     nn.ReLU(...)
//   )
//
// def forward(self, x):
//   return self.double_conv(x)
//     ```
//   ])
// ]
//
// #let b2 = std-block(breakable: false)[
//   #box(
//     fill: Up_colour,
//     outset: 1mm,
//     inset: 0em,
//     radius: 3pt,
//   )[#text(white, size: 12pt, font: "JetBrainsMono NFM")[Up]] \
//   #listing(line-numbering: none, [
//     ```python
// class Up(nn.Module):
//   def __init__(self, in_c, out_c, 
//                      bilinear=True):
//       super(Up, self).__init__()
//     if bilinear:
//       self.up = nn.Upsample(...)
//     else:
//       self.up = nn.ConvTranspose2d(...)
//     self.conv = DoubleConv(...)
//  
//   def forward(self, x1, x2):
//     x1 = self.up(x1)
//     x1 = F.pad(x1, ...)
//     x = torch.cat([x2, x1], dim=1)
//     return self.conv(x)
//     ```]
//     )
// ]
//
// #let b3 = std-block(breakable: false)[
//   #box(
//     fill: OutConv_colour,
//     outset: 1mm,
//     inset: 0em,
//     radius: 3pt,
//   )[#text(white, size: 12pt, font: "JetBrainsMono NFM")[OutConv]] \
//   #listing(line-numbering: none, [
//     ```python
// class OutConv(nn.Module):
// def __init__(self, in_c, out_c):
//   super(OutConv, self).__init__()
//   self.conv = nn.Conv2d(...)
//
// def forward(self, x):
//   return self.conv(x)
//     ```
//   ])
// ]
//
// #let b4 = std-block(breakable: false)[
//   #box(
//     fill: Down_colour,
//     outset: 1mm,
//     inset: 0em,
//     radius: 3pt,
//   )[#text(white, size: 12pt, font: "JetBrainsMono NFM")[Down]] \
//   #listing([
//     ```python
// class Down(nn.Module):
//   def __init__(self, in_c, out_c):
//     super(Down, self).__init__()
//     self.maxpool_conv = nn.Sequential(
//       nn.MaxPool2d(...),
//       DoubleConv(...)
//     )
//  
//   def forward(self, x):
//     return self.maxpool_conv(x)
//     ```
//     ]
//   )
// ]
//
// #grid(
//   columns: (1fr, 1fr),
//   column-gutter: 2mm,
//   inset: 0mm,
//   [#b1 #v(-2.5mm) #b4], [#b2 #v(-2.5mm) #b3]
// )

The U-Net architecture was released in a landmark paper 2015, having since garnered over 100,000 citations. U-Net gets its very literal name from its architecture, which resembles a wide letter U. The U-Net architecture is classed as a #acr("FCN"), which is a type of network that is particularly effective for image segmentation tasks. 

The architecture consists of two main parts: the encoder and the decoder. Referring to @fig:unet, the encoder part of the network is the left-hand side portion down to the bottleneck, with the decoder being the following layers. First, the encoder increases the number of channels while decreasing the spatial dimensions of the input image. This is done in layers. The first layer is a double convolutional layer #ball(DoubleConv_colour), which consists of two convolutional layers, each followed by a batch normalization and ReLU activation. This is followed by a max pooling layer #ball(Down_colour), which reduces the spatial dimensions of the image. The second layer is another double convolutional layer #ball(DoubleConv_colour), which again increases the number of channels while decreasing the spatial dimensions. This process continues until the bottleneck is reached.

The decoder then reverses this process. After the bottleneck, the decoder begins with an upsampling layer #ball(Up_colour), which increases the spatial dimensions of the image again. This is followed by a concatenation with the corresponding encoder layer, which allows the model to retain spatial information lost during the downsampling process. Skip connections are commonly used in various architectures, as they help to retain spatial information. Following the concatenation, another double convolutional layer #ball(DoubleConv_colour) is applied, which reduces the number of channels. This process continues until the final output layer is reached. In the final layer, a double convolution is initially applied after the concatenation, followed by a final convolutional layer #ball(OutConv_colour). This final convolutional layer reduces the number of channels to the number of classes in the segmentation task. The final output is a segmentation map, which indicates the predicted class for each pixel in the input image.

#let fig1 = { image("../../../../figures/img/models/unet/unet.png") }

#std-block(breakable: false)[
  #figure( fig1,
  caption: [U-Net Architecture.]
  ) <fig:unet>
]

This architecture significantly advanced the field of image segmentation. One of its key innovations is its extensive use of skip connections, which directly fuse the contextual information from the encoder with the spatial detail from the decoder. This mechanism helps the model recover fine-grained spatial features that are often lost during the downsampling process. Additionally, U-Net employs double convolutional layers rather than single convolutions. This structure allows the model to extract more complex and abstract features from the input data. The architecture is also fully convolutional, meaning it avoids fully connected layers altogether. As a result, U-Net can be applied to images of varying sizes without needing architectural modifications, making it highly versatile across different datasets and tasks.

U-Net was originally introduced to address the challenge of limited labelled data in the medical imaging field, where precise segmentation is critical and annotated samples are scarce. However, its effectiveness has extended far beyond this initial scope. It has inspired a wide range of variants, including U-Net++, Attention U-Net, ResUNet, and Mobile U-Net, each adapted for specific applications or resource constraints.

Despite its strengths, U-Net is not without limitations. It can be memory-intensive and computationally demanding, particularly due to the skip connections. These connections require storing high-resolution feature maps from earlier layers in memory, which increases the model's footprint and can limit scalability. Nevertheless, U-Net's widespread success, ability to generalize well on small datasets, and strong performance in segmentation tasks make it an ideal candidate for this project.

For this project, the U-Net architecture was implemented using the implementation by milesial#footnote([Their github repository is found here: https://github.com/milesial/Pytorch-UNet with the implemented version found in this project's GitLab repository.]) with minor modifications. This implementation features a modular design with clearly defined components for the encoder and decoder paths. Each downsampling step comprises a max-pooling layer followed by a double convolutional block, while each upsampling step includes either bilinear interpolation or transposed convolution, concatenation with the corresponding encoder feature map, and another double convolutional block. The model supports flexibility in input image sizes due to its fully convolutional nature. Training routines are provided as well, but new ones were ultimately implemented for easier switching between this project's models.

In this project, U-Net is the simplest of the models explored for the task of predicting paths through an intersection based solely on satellite imagery. Rather than serving as a segmentation backbone or part of a larger system, the U-Net model is trained end-to-end to directly output the predicted traversal path. While U-Net is traditionally used for semantic segmentation tasks, its encoder-decoder structure and spatial information retention via skip connections make it a suitable candidate for this kind of spatially grounded prediction. It is acknowledged, however, that U-Net alone is unlikely to yield highly precise results in such a complex task. Nonetheless, its simplicity and proven performance on pixel-wise prediction problems make it a valuable baseline for evaluating how well standard architectures can handle intersection-based path planning.