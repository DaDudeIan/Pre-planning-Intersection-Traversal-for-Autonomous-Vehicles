#import "../../../../lib/mod.typ": *
=== U-Net <c4:unet>

#let DoubleConv_colour = rgb("7EA6E0")
#let Down_colour = rgb("FF3333")
#let Up_colour = rgb("66CC00")
#let OutConv_colour = rgb("FFB366")

Hello #ball(DoubleConv_colour) #ball(Down_colour) #ball(Up_colour) #ball(OutConv_colour)

#let b1 = std-block(breakable: false)[
  #box(
    fill: DoubleConv_colour,
    outset: 1mm,
    inset: 0em,
    radius: 3pt,
  )[#text(white, size: 12pt, font: "JetBrainsMono NFM")[DoubleConv]] \
  #listing(line-numbering: none, [
    ```python
class DoubleConv(nn.Module):
def __init__(self, in_c, out_c):
  super(DoubleConv, self).__init__()
  self.double_conv = nn.Sequential(
    nn.Conv2d(...),
    nn.BatchNorm2d(...),
    nn.ReLU(...),
    nn.Conv2d(...),
    nn.BatchNorm2d(...),
    nn.ReLU(...)
  )

def forward(self, x):
  return self.double_conv(x)
    ```
  ])
]

#let b2 = std-block(breakable: false)[
  #box(
    fill: Up_colour,
    outset: 1mm,
    inset: 0em,
    radius: 3pt,
  )[#text(white, size: 12pt, font: "JetBrainsMono NFM")[Up]] \
  #listing(line-numbering: none, [
    ```python
class Up(nn.Module):
  def __init__(self, in_c, out_c, 
                     bilinear=True):
      super(Up, self).__init__()
    if bilinear:
      self.up = nn.Upsample(...)
    else:
      self.up = nn.ConvTranspose2d(...)
    self.conv = DoubleConv(...)
  
  def forward(self, x1, x2):
    x1 = self.up(x1)
    x1 = F.pad(x1, ...)
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)
    ```]
    )
]

#let b3 = std-block(breakable: false)[
  #box(
    fill: OutConv_colour,
    outset: 1mm,
    inset: 0em,
    radius: 3pt,
  )[#text(white, size: 12pt, font: "JetBrainsMono NFM")[OutConv]] \
  #listing(line-numbering: none, [
    ```python
class OutConv(nn.Module):
def __init__(self, in_c, out_c):
  super(OutConv, self).__init__()
  self.conv = nn.Conv2d(...)

def forward(self, x):
  return self.conv(x)
    ```
  ])
]

#let b4 = std-block(breakable: false)[
  #box(
    fill: Down_colour,
    outset: 1mm,
    inset: 0em,
    radius: 3pt,
  )[#text(white, size: 12pt, font: "JetBrainsMono NFM")[Down]] \
  #listing([
    ```python
class Down(nn.Module):
  def __init__(self, in_c, out_c):
    super(Down, self).__init__()
    self.maxpool_conv = nn.Sequential(
      nn.MaxPool2d(...),
      DoubleConv(...)
    )
  
  def forward(self, x):
    return self.maxpool_conv(x)
    ```
    ]
  )
]

#grid(
  columns: (1fr, 1fr),
  column-gutter: 2mm,
  inset: 0mm,
  [#b1 #v(-2.5mm) #b4], [#b2 #v(-2.5mm) #b3]
)

#let fig1 = { image("../../../../figures/img/models/unet/unet.png") }

#std-block(breakable: false)[
  #v(-1em)
  #box(
    fill: theme.sapphire,
    outset: 0em,
    inset: 0em,
  )
  #figure( fig1,
  caption: [U-Net Architecture.]
) <fig.unet>
]