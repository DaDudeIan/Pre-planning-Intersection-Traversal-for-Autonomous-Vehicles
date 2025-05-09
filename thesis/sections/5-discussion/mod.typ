#import "../../lib/mod.typ": *
= Discussion <c6:Discussion>

In this section...

== Integration with existing systems <c6:integration>

== Shortcomings <c6:shortcomings>
// Notable shortcomings of project: requires latest satellite images to be completely useful. (like contruction zones, etc)
// comment on branching loss not working well with thick line (comp9)

//== Future Work <c6:future-work>



== Other considerations <c6:other-considerations>
// Use skeletonization to reduce the number of points in the line and make it 1 pixel wide
// other considered languages: C/C++, Fortran
// other models: RL
// GPS not enough

// thought experiment on how to implement for other domains (warehouse robots, racing, overall AV, deep sea automation). speculative

// examiner from the production engineering world

// How to find the angle to rotate? When driving with a GPS, use coordinates to the centre. Find angle by finding the angle between the two points. 

== Ablation <c6:ablation>
MAYBE, if time allows

Hyperparameters, scheduler (cosann vs exp), optimizer (adam vs adamw), weight initialization