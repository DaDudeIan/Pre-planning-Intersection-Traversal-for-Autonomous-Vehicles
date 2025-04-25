#import "../../lib/mod.typ": *

== Research Questions <c1:research_questions>

// Waypoint-based vs subset of pixels-based path drawing

// What kind of model generate the best result? (RL, Diffusion, CNN, etc.)

// Something regarding loss function: Is it possible to create a loss function that captures likenesses between two paths?


// potential if all goes well: How do we ensure that the vehicle follows the path correctly? (localization)


The following research questions have been formulated to address key challenges in #acr("AV") path planning at intersections. The questions are designed to explore the effectiveness of different approaches and models in generating accurate and efficient paths for autonomous vehicles. The research questions are as follows:

#[
  #set par(first-line-indent: 0em)
  #set enum(numbering: req-enum.with(prefix: "RQ-", color: theme.teal))
  + How can pixel-subset-based deep learning approaches be optimized to improve accuracy and efficiency in path planning for autonomous vehicles at intersections? How do convolution-based and transformer-based models compare in this context?
  + Is it possible to design a loss function that effectively captures the similarity between generated and desired paths for autonomous vehicles without forcing exact matches?
  + Is it possible to create a dataset that allows for the training of a model, such that the data is not too stringent to a singular path?
  //+ What strategies can be employed to ensure robust localization of autonomous vehicles while following generated paths?  
]