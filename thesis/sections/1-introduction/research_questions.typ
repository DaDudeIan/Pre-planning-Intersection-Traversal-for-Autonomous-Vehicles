#import "../../lib/mod.typ": *

== Research Questions <c1:research_questions>

// Waypoint-based vs subset of pixels-based path drawing

// What kind of model generate the best result? (RL, Diffusion, CNN, etc.)

// Something regarding loss function: Is it possible to create a loss function that captures likenesses between two paths?


// potential if all goes well: How do we ensure that the vehicle follows the path correctly? (localization)

#[
  #set par(first-line-indent: 0em)
  #set enum(numbering: req-enum.with(prefix: "RQ-", color: theme.teal))
  + How does a waypoint-based approach compare to a pixel-subset-based approach in terms of accuracy and efficiency in path planning for autonomous vehicles at intersections?
  + Which type of deep learning model produces the most optimal results for pre-planned intersection traversal?
  + Is it possible to design a loss function that effectively captures the similarity between generated and optimal paths for autonomous vehicles without it only allowing for exact paths?
  + Is it possible to create a dataset that allows for the training of a model, such that the data is not too stringent to a singular path?
  //+ What strategies can be employed to ensure robust localization of autonomous vehicles while following generated paths?  
]

