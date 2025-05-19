#import "../../../../lib/mod.typ": *

=== Continuity Loss #checked <c4:topology_loss>

// @topo1 also highlights the use of BCE in combination with a topology loss.
// Mention TopoLoss from TopoNets, but that the smoothing is not what is needed here.
// 

// As the previous sections have highlighted, there is a dire need for a topology-based loss function. The cold map loss and the BCE loss are both excellent at penalizing paths that are far from the true path, but they do not penalize breaks in the path or ensure that the predicted path is continuous. This is where the topology loss comes in. The topology loss function will revolve around getting the output from the model to be a singular, continuous components, meaning it does not contain breaks or holes. Formally, this is done by aiming for specific Betti number values, ensuring that the predicted path has a single connected component and no loops. The details of this approach are discussed in the following sections.

The second of the topology based loss functions is the continuity loss function. The continuity loss function will revolve around getting the output from the model to be a singular, continuous component, meaning it does not contain breaks or holes. Formally, this is done by aiming for specific Betti number values, ensuring that the predicted path has a single connected component and no loops. The details of this approach are discussed in the following sections.

Considerations of using existing topology existing methods. Dep #etal @topoloss introduced TopoNets and TopoLoss. This loss function revolves around penalizing jagged paths and encouraging smooth, brain-like topographic organization within neural networks by reshaping weight matrices into two-dimensional cortical sheets and maximizing the cosine similarity between these sheets and their blurred versions. Cortical sheets are two-dimensional grids formed by reshaping neural network weight matrices to emulate the brain's spatial organization of neurons, enabling topographic processing. While initially interesting in the context of this project, simple testing showed that the values returned from this loss, did not give a proper presentation of the path's topology, outside of its smoothness. And while smoothness is a part of the topology, this will largely be handled by the #acr("CE") loss.

This section will present the topology-based loss function designed for this project, with a focus on ensuring that the predicted path is continuous and does not contain any breaks or holes. This is a crucial aspect of the task at hand, as the goal is to create a path that a vehicle can follow. Breaks in a path would be unrealistic for a grounded vehicle to follow. As a starting point, it is important to understand the concept of Betti numbers:

#std-block(breakable: true)[
  #box(
    fill: theme.sapphire.lighten(10%),
    outset: 1mm,
    inset: 0em,
    radius: 3pt,
  )[#text(white, size: 12pt, font: "JetBrainsMono NFM")[Betti Numbers]] \
  Betti numbers @betti come from algebraic topology, and are used to distinguish topological spaces based on the connectivity of $n$-dimensional simplicial complexes. The $n$th Betti number, $beta_n$, counts the number of $n$-dimensional holes in a topological space. The Betti numbers for the first three dimensions are:
   - $beta_0$: The number of connected components.
   - $beta_1$: The number of loops.
   - $beta_2$: The number of voids.
   The logic follows that, in 1D, counting loops are not possible, as it is simply a line. This, if the number is greater than 1, means it is split into more than one component. In 2D, the number of loops is counted, i.e. a circled number of pixels. In 3D, this extends to voids. 
 ]

With this, for the 2D images used in this project, the Betti numbers are $beta_0$ and $beta_1$. The continuity loss is designed to ensure that the predicted path has a single connected component and no loops. This is achieved by aiming for the Betti numbers to be $beta_0 = 1$ and $beta_1 = 0$. Higher dimensional Betti numbers are not relevant for this project, as the images are 2D. While Betti numbers are a powerful tool for topology analysis, they are not directly applicable to the loss function as they are discrete values. This means that they offer no gradient information, which is essential for training a neural network. Instead, persistent homology is deployed.

#acr("PH") is a mathematical tool used to study topological features of data. Homology itself is a branch of algebraic topology concerned with procedures to compute the topological features of objects. Persistent homology extends the basic idea of homology by considering not just a single snapshot of a topological space but a whole family of spaces built at different scales. Instead of calculating Betti numbers for one fixed space, a filtration is performed. This filtration is a sequence of spaces, where each space is a subset of the next, i.e. a nested sequence of spaces where each one is built by gradually growing the features by some threshold. As this threshold varies, topological features such as connected components and loops will appear (be born) and eventually merge or vanish (die).

This birth and death of features is recorded in what is known as a persistence diagram or barcode (See @fig:persistent_homology). In these diagrams, each feature is represented by a bar (or a point in the diagram) whose length indicates how persistent, or significant, the feature is across different scales. Features with longer lifespans are generally considered to be more robust and representative of the underlying structure of the data, whereas those that quickly appear and disappear might be attributed to noise.
// // possibility of exploding loss when noisy image


#let fig1 = { image("../../../../figures/img/loss_example/comp12.png", width: 80%) }
#let fig2 = { image("../../../../figures/img/loss_example/pers_bar.png") }
#let fig3 = { image("../../../../figures/img/loss_example/pers_diag.png") }

// #let fig4 = { image("../../../../figures/img/loss_example/comp7.png", width: 80%) }
// #let fig5 = { image("../../../../figures/img/loss_example/pers_bar_2.png") }
// #let fig6 = { image("../../../../figures/img/loss_example/pers_diag_2.png") }

#std-block(breakable: false)[
  #figure(
    grid(
    columns: (1fr, 1fr, 1fr),
    column-gutter: 1mm,
    align: (center, center),
    fig1, fig2, fig3,
    //fig4, fig5, fig6
    ),
    caption: [The top row shows a connected path along with its persistence barcode and persistence diagram, while the bottom row shows a disconnected path. The number of lines in the barcode, stems from the fact that the images are rather large in size and thus the number of built spaces are many.]
  ) <fig:persistent_homology>
]

The following will cover the method used to achieve the continuity loss function. It closely follows the work done by Clough #etal @topology_loss with some minor changes that will be pointed out. Furthermore, the implementation is done using the Gudhi library, which is a Python library for computational topology. It provides a set of tools for computing persistent homology and other topological features of data. The library is designed to be efficient and easy to use, making it a good choice for this project. Lastly, PyTorch's ability to create custom autograd functions is used to implement the persistent homology loss function.

#let fig1 = { image("../../../../figures/img/loss_example/clough_res.png") }

#let fig = std-block(breakable: false)[
  #figure(
    fig1,
    caption: [Fig. 6 from @topology_loss.]
  ) <fig:clough_res>
]

#let c = [#h(4mm)Before presenting the loss function definition, it is important to understand exactly what this loss function is supposed to achieve. The intuition behind the loss function is, as mentioned, to drive the network towards achieving a specified set of betti numbers. @fig:clough_res comes from the work done by Clough #etal, where they show the result of using three different topological priors, i.e. the betti numbers. In all cases shown, it drives the output towards the desired number of components and loops. Therefore, defining the topological priors of the function that is to train the models as $beta_0 = 1, beta_1 = 0$, should yield results similar to the top-most output image in @fig:clough_res. As they point out, however, this loss function does not ensure correct looking outputs, since topology alone is not enough to describe the actual shape of an output. Therefore, in testing it will only be shown in combination with the CE loss function.]


#wrap-content(fig, c, align: right, columns: (3fr, 2.5fr))

#h(4mm) Now, the output from the network is a tensor of size 
$ Omega = H times W $ 
where, in this project, $H=400$ and $W=400$. Then for each pixel $x in Omega$, the network outputs a logit vector:
$ L(x) in RR^C $
where $C$ is the number of classes. A softmax function is applied to give class probabilities:
$
  P_c (x)  = (e^(L_c(x)))/(sum_(j=1)^C e^(L_j(x)))
$

Here it is noted that the background class is $"bg" = 0$. Following that, a foreground probability map is created by removing the background probabilities:
$
  "fg"(x) = 1 - P_"bg" (x) in [0,1]
$  

Now, persistent homology on images is usually computed through super-level sets. Clough #etal formulates PH on super-level sets, which is the opposite of sub-level sets. Sub-level sets were chosen, as the alternative wouldn't work in practice, as Gudhi is kept in sub-level mode. Therefore, an inverted version of the foreground probability map is created:
$
  f(x) = 1 - "fg"(x) in [0,1]
$
where low values mark confident foreground pixels, and high values mark confident background pixels. The filtration, as generated inside Gudhi, is defined as 
$
  K_0 subset.eq K_(t_1) subset.eq dots.h.c subset.eq K_1
$
where each sub-level set is
$
  K_t = {x in Omega | f(x) <= t}
$ 
This happens inside the `CubicalComplex` function, which is a Gudhi function that creates a cubical complex from the input. In short, the cubical complex is a data structure that represents the topological features of the input data. The Gudhi library is used to compute the persistent homology of the cubical complex, which is then used to calculate the Betti numbers. 

From the cubical complex, persistence pairs are found, which are the birth and death values of the topological features. Gudhi returns a multiset for each homology dimension:
$
  D_k (f) = {(b_i^((k)), d_i^((k))) }
$
with the constraint $0 <= b_i^((k)) < d_i^((k)) <= infinity$, where $k=0$ for connected components and $k=1$ for loops. The component whose death time is $+infinity$ is discarded, as it is not influenced by the other pixels' values and all other pairs have finite deaths.

At this stage, some extra work is required, as Gudhi also returns _where_ the feature pairs are born and die. In other words, flat Fortran-order indices of the pixels whose grey-values equal the birth and death values. These indices are remapped to PyTorch's row-major order with a small helper function:
$
  "_F2C"(n) = r + c H
$
with $(r, c) = "divmod"(n, H)$ where $"divmod"$ returns the element-wise quotient and remainder of the two inputs. 

Finally, the actual loss is found by defining the persistence, ignoring bars that are shorter than some threshold $epsilon$:
$
  "pers"_i^((k)) = d_i^((k)) - b_i^((k)) > epsilon
$
where the threshold $epsilon$ is typically a low value. Thus, the per-image loss, i.e. each image in a batch, is defined as:
$
  cal(L)(f) = w_0 sum_((b,d) in D_0(f), d-b > epsilon) (d-b)^p + w_1 sum_((b,d) in D_1(f), d-b > epsilon) (d-b)^p
$ <eq:loss_topo>
where $w_0$ and $w_1$ are weights for the two dimensions, and $p$ is a power parameter. The loss is then averaged over the batch size:
$
  "Loss" = 1/B sum_(b=1)^B cal(L)(f_b)
$
Because each summand in @eq:loss_topo is built from pixel values $f(x)$, gradients propagate exactly to those birth- and death-pixels:

$
  partial / (partial f(x)) (d-b)^p = cases(
    #align(right)[#box[$-p(d-b)^(p-1)$], #h(5mm)]& "if" x "is the birth pixel,",
    #align(right)[#box[$p(d-b)^(p-1)$],]& "if" x "is the death pixel,",
    #align(right)[#box[$0$],]& "otherwise"
  )
$
This comes from the fact that the partial derivatives are immediate, as the summand for any bar depends only on the two scalar values from $f$.

Examples of the PH-based loss function is shown in @fig:loss_topo_cont. The far left image shows the output from a model, showing clear and separated components. This is the highly undesired trait this loss is designed to penalize. The loss value is $1.35$, which is a very high value. The next image shows the same model after a backwards pass has been made by the loss function. Already, it is clear to see that the model has improved, as the components are now largely connected, with only some stragglers. Then after another two iterations, the model is nearing a loss of $0$. The last image has a near $0$ loss after just another iteration. 

This highlights the effectiveness of this loss function. If this keeps going, the model will eventually reach a loss of $0$. However, this rudimentary testing of the loss function also highlights the fact that it alone does not care for class labels. After just 20 iterations, it would label every pixel as the same class, as it is only concerned with the topology of the path. This is a trait of the loss function that is not desired, but when it works in conjunction with the other loss functions, it gets balanced to create more accurate results. This will be covered in depth in @c4:training-strategy where the training strategy is presented.

#let fig1 = { image("../../../../figures/img/loss_example/output0.png") }
#let fig2 = { image("../../../../figures/img/loss_example/output1.png") }
#let fig3 = { image("../../../../figures/img/loss_example/output2.png") }
#let fig4 = { image("../../../../figures/img/loss_example/output3.png") }
#let fig5 = { image("../../../../figures/img/loss_example/output4.png") }

#std-block(breakable: false)[
  #figure(
    grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      fig1, fig2, fig4, fig5
    ),
  caption: [ Results of refining trained model a few iterations on the same image. ]
  ) <fig:loss_topo_cont>
]

Finally, before any loss function can be used, there needs to exist a dataset to train on. The dataset needs to contain content that it is desired for the model to learn. Furthermore, it is important for each entry to have their annotated ground truth, otherwise the loss function would not have anything to compare against. This is especially the case with models trained on supervised learning. Thus, the next section will cover the dataset created for this project.