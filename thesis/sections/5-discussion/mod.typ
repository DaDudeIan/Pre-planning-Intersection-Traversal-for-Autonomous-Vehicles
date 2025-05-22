#import "../../lib/mod.typ": *
= Discussion <c6:Discussion>

In this section...

== Integration with existing systems <c6:integration>

== Shortcomings <c6:shortcomings>
// Notable shortcomings of project: requires latest satellite images to be completely useful. (like construction zones, etc)

//== Future Work <c6:future-work>



== Other considerations <c6:other-considerations>
// Use skeletonization to reduce the number of points in the line and make it 1 pixel wide
// other considered languages: C/C++, Fortran
// other models: RL
// GPS not enough

// thought experiment on how to implement for other domains (warehouse robots, racing, overall AV, deep sea automation). speculative

// examiner from the production engineering world

// How to find the angle to rotate? When driving with a GPS, use coordinates to the centre. Find angle by finding the angle between the two points. 

// on-board vs cloud processing

// Look into how ViT and Swin handle structural priors

// Discuss how a model's loss can sky rocket but the accuracy can remain good.

// Why does the transformer models appear to play less nice with continuity loss function?

// if interesting results, look at the mainly cmap trained model (deeplab_cmap-ce opposed to ce-cmap)

// Longer training test:
// 1000 epochs
// Overall mIoU: 0.4545
// Per class mIoU: [0.9767, 0.3483, 0.3379, 0.3409, 0.2688]
// 5000 epochs
// Overall mIoU: 0.4501
// Per class mIoU: [0.9758, 0.3485, 0.337, 0.3313, 0.258]

== Ablation <c6:ablation>
MAYBE, if time allows

Hyperparameters, scheduler (cosann vs exp), optimizer (adam vs adamw), weight initialization