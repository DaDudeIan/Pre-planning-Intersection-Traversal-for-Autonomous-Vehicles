#import "../../../lib/mod.typ": *
== Dataset Creation <c4:data>

The creation of a proper dataset is crucial for making sure the model learns the task desired for it to perform. The dataset will have to work hand-in-hand with the model architecture and the loss function to ensure that the model learns the task effectively. Many aspects are to be considered when creating a dataset for a task as specific as this project sets out to create:
- It should be large enough to capture the complexity of the task. Size can be artificially increased through data augmentation.
- It should be diverse enough to capture the variety of scenarios that can occur at an intersection.
- It should allow for some leniency when it comes to generating paths, as the model should not be too stringent to a singular path.
- For the purposes of this project, its creation should seek to answer #RQ(4) by providing a dataset that allows for the training of a model that can generate paths that are not too stringent to a singular path. 

=== Cold maps <c4:cold_maps>

The deduced method for training the model, as detailed in @c4:loss, revolves around the use of a cold map. A cold map representation of the desired path was chosen for a small simplification in the loss function. It penalizes points that are further from the desired path, and does not do this for points that are on the path. Creating this cold map was done in several steps. First, a grid of the same size as the input image is created. The input image is the path drawn in white on a black background, as shown in centre @fig.dataset_example. This means that the only occupied pixels are those taken up by the path. In this grid, the coordinates of the closest non-zero pixel is found by iterating over the entire input image containing the path. The complexity of this operation will be covered in the following sections. Next, the distance between the current pixel and the closest non-zero pixel is calculated. This distance is then compared to a threshold value to determine its value. If it is further away, the resulting penalty from the loss function should be higher. Different values for the threshold and the exponent of the distance calculation were tested to find the best combination. Lastly, the cold map is saved in a structured folder format for later use in training. Later, the created data is put through augmentation to inflate the size of the dataset and increase its diversity.

==== Finding the distance to the desired path <c4:cold_maps.dist>

The algorithm for finding the distance to the closest point on the desire path is shown in @code.distance_grid. 

#listing([
  ```python
  nearest_coords = np.zeros((path.shape[0], path.shape[1], 2), dtype=int)
  for i in range(path.shape[0]):
    for j in range(path.shape[1]):
      if binary[i, j] == 0:
        min_dist = float('inf')
        nearest_coord = (i, j)
        for x in range(path.shape[0]):
          for y in range(path.shape[1]):
            if binary[x, y] != 0:
              dist = hypot(i - x, j - y)
              if dist < min_dist:
                min_dist = dist
                nearest_coord = (x, y)
        nearest_coords[i, j] = nearest_coord
      else:
        nearest_coords[i, j] = (i, j)
  
  ```
],
caption: [Non-parallelized code for finding the nearest point on the path.]
) <code.distance_grid>

The algorithm in @code.distance_grid iterates over every pixel of the aforementioned grid. For each point in the grid, the first thing checked is the value of a binary map `binary`, which is also a grid of the same size as the input image, if there is something there. This is done to avoid calculating the distance for points that are already on the path and are simply assigned their current coordinates. If the point is not on the path, i.e. it as a value of 0, the algorithm iterates through the entire path image, and if it encounters a non-zero value, it calculates the distance between the current point and the point on the path. There is no guarantee that this is the closest point however, so the algorithm saves the coordinates of the found point in a variable if it currently is the closest point. Finally, the closest point at the end is saved to that grid entry's coordinates. This is then repeated for every single point in the grid until every point has been assigned the coordinates of the closest point on the path. This grid will later be used under the name `coords`.

The shown algorithm is not parallelized and has a complexity of $cal(O)(n^4)$, where $n$ is the size of the input image. This is due to the nested loops that iterate over the entire image to find the closest point. This operation is done for every pixel in the image, resulting in a high complexity. The actual implementation of this algorithm is parallelized, but the non-parallelized form is shown here. While the parallelized version is significantly faster by assigning each core its own chunk of the data, going from 73 minutes on a $400 times 400$ down to 8 minutes on an 8-core CPU, the complexity remains the same. Further improvements could be made both to the complexity of the implementation and parallelization could be distributed to a GPU or the cloud for even faster computation, but this remains future work.


==== Creating the cold map <c4:cold_maps.create>

To start the creation of the cold map, a distance grid is created using Pythagoras' theorem between the coordinates of the point of the grid and the coordinates saved within, retrieved from the aforementioned `coords` variable. A masking grid is then created by comparing the distance grid to a threshold value. This results in each grid point being calculated using:

$
d_(i j) = sqrt((i - c_(i j 0))^2 + (j-c_(i j 1))^2)
$ <eq:distance>

$ d t_(i j) = cases(
  #align(right)[#box[$d_(i j)$]]& "if" d_(i j) < t,
  #align(right)[#box[$t + (d_(i j) - t)^e$]]& "otherwise"
) $

where $c$ = `coords`, $c_(i j 0)$ = `coords[i, j][0]`, $t$ is the threshold value, and $e$ is the exponent value. All three of these can be seen as function parameters in the function declaration in @code.coldmap. The distance grid is then normalized to a range of 0 to 255 to minimize space usage such that it fits within a byte, i.e. an unsigned 8-bit integer. This is done by subtracting the minimum value and dividing by the range of the values. The resulting grid is then saved as a cold map. The resulting cold map can be seen in the rightmost image in @fig.dataset_example.





#listing([
  ```python
def coords_to_coldmap(coords, threshold: float, exponent: float):
  rows, cols = coords.shape[0], coords.shape[1]

  distances = np.zeros((rows, cols), dtype=np.float32)
  for i in range(rows):
    for j in range(cols):
      distances[i, j] = hypot(i - coords[i, j][0], j - coords[i, j][1])
  
  distances_c = distances.copy()
  mask = distances > threshold
  distances_c[mask] = threshold + (distances[mask] - threshold) ** exponent
  
  distances_c_normalized = 255 * (distances_c - distances_c.min()) / (distances_c.max() - distances_c.min())
  
  return distances_c_normalized.astype(np.uint8)
  ```
],
caption: [Non-parallelized code for finding the nearest point on the path.]
) <code.coldmap>



To figure out the optimal values for the threshold and exponent, a grid search was performed. The grid search was done by iterating over a range of values for both the threshold and the exponent. The resulting cold maps were then evaluated by a human to determine which combination of values resulted in the most visually appealing cold map. For a $400 times 400$ image, the optimal values were found to be $t=20$ and $e=0.5$. The grid of results can be seen in figure @fig.coldmaps_comp. In testing, the value of $e=1$ was excluded as it had no effect on the gradient produced in the cold map, meaning all values of $t$ produced the same map.

// #std-block(breakable: false)[
//   #v(-1em)
//   #box(
//     fill: theme.sapphire,
//     outset: 0em,
//     inset: 0em,
//   )
//   #figure(
//   grid(
//   columns: (1fr),
//   column-gutter: 1mm,
//   align: (horizon, center),
//     image("../../../figures/img/dataset_example/coldmaps_comp.png"),
//   image("../../../figures/img/dataset_example/inferno_coldmap.png", width: 90%)
// ),
//   caption: [Results of testing threshold values $t in {10, 20, 40, 70, 100}$ and exponent values $e in {0.5, 0.75, 1.25}$. The colour map is shown beneath the results#footnote([The colour map is retrieved from the matplotlib docs: https://matplotlib.org/stable/users/explain/colors/colormaps.html]).]
// ) //<fig.coldmaps_comp>
// ]

#let my_brace = {v(2mm); layout(size => {$overbrace(#h(size.width - 4pt))$})}
#let threshold_text(n) = {set align(center); set text(size: 9pt); $t = $ + " " + str(n)}
#let my_braces = {let s = 625%; stack(dir: ttb, spacing: 1.2em, v(0.3em), $lr("{", size: #s)$, $lr("{", size: #s)$,$lr("{", size: #s)$)}
#let e_stack = {set text(size: 9pt); stack(dir: ttb, spacing: 5em, (1cm), rotate(270deg, reflow: true, $e=0.5$), rotate(270deg, reflow: true, $e=0.75$), rotate(270deg, reflow: true, $e=1.25$))}

#std-block(breakable: false)[
  #v(-1em)
  #box(
    fill: theme.sapphire,
    outset: 0em,
    inset: 0em,
  )
  #grid(
    columns: (auto, auto, auto, 1fr),
    column-gutter: 1mm,
    [], [], [],
    grid(
      columns: (-1mm, 1fr),
      [],
      grid(
        columns: (1fr, 1fr, 1fr, 1fr, 1fr),
        column-gutter: 1mm,
        row-gutter: -1mm,
        [], [], h(2.5mm) + [Thresholds] + v(3mm), [], [],
        threshold_text(10), threshold_text(20), threshold_text(40), threshold_text(70), threshold_text(100),
        {my_brace}, {my_brace}, {my_brace}, {my_brace}, {my_brace},
      )
    )
    ,
    {set align(horizon); v(-0.75cm); h(4mm); rotate(270deg, reflow: true, [Exponent])}, e_stack, my_braces, 
      v(-2mm) + image("../../../figures/img/dataset_example/coldmaps_comp_notitles.png"),
    grid.cell(
      colspan: 4,
      v(-4mm) +
        [#figure(
          image("../../../figures/img/dataset_example/inferno_coldmap.png", width: 100%) + v(-3mm),
          caption: [Results of testing threshold values $t in {10, 20, 40, 70, 100}$ and exponent values $e in {0.5, 0.75, 1.25}$. The colour map is shown beneath the results#footnote([The colour map is retrieved from the matplotlib docs: https://matplotlib.org/stable/users/explain/colors/colormaps.html]).]
      ) <fig.coldmaps_comp> ]
    )
  )
]


Finally, a comparison between the retrieved satellite image of an intersection, the optimal path through it, and the cold map generated by the process described above are shown in @fig.dataset_example. This highlights the importance of the cold map in the training process as opposed to the single line path. The cold map allows for a more lenient path to be generated, as the model is not penalized for deviating slightly from the path. 


#let fig1 = { image("../../../figures/img/dataset_example/map.png") }
#let fig2 = { image("../../../figures/img/dataset_example/map_path_thin.png") }
#let fig3 = { image("../../../figures/img/dataset_example/cold_map.png") }

#std-block(breakable: false)[
  #v(-1em)
  #box(
    fill: theme.sapphire,
    outset: 0em,
    inset: 0em,
  )
  #figure(
  grid(
  columns: (1fr, 1fr, 1fr),
  column-gutter: 1mm,
  align: (horizon, center),
  fig1,fig2,fig3
  
),
caption: [Example of satellite image, next to the desired path through with. To the far right, the generated cold map is shown with threshold $t=20$ and exponent $e=0.5$. Notice how it is only the points very close to the path that are very cold, while the rest of the map is warmer the further away it is.]
) <fig.dataset_example>
]

=== Data Augmentation <c3:data_augmentation>


=== Dataset Structure <c3:dataset_structure>