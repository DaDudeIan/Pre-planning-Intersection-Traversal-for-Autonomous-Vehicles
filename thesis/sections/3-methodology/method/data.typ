#import "../../../lib/mod.typ": *
== Dataset Creation <c4:data>

The creation of a proper dataset is crucial for making sure the model learns the task desired for it to perform. The dataset will have to work hand-in-hand with the model architecture and the loss function to ensure that the model learns the task effectively. Many aspects are to be considered when creating a dataset for a task as specific as this project sets out to create:
- It should be large enough to capture the complexity of the task. Size can be artificially increased through data augmentation.
- It should be diverse enough to capture the variety of scenarios that can occur at an intersection.
- It should allow for some leniency when it comes to generating paths, as the model should not be too stringent to a singular path.
- For the purposes of this project, its creation should seek to answer #RQ(dataset_rq) by providing a dataset that allows for the training of a model that can generate paths that are not too stringent to a singular path. 

=== Cold maps <c4:cold_maps>

The deduced method for training the model, as detailed in @c4:loss, includes the use of a cold map. A cold map representation of the desired path was chosen for a small simplification in the loss function. It penalizes points that are further from the desired path, and does not do this for points that are on the path. Creating this cold map was done in several steps. First, a grid of the same size as the input image is created. The input image is the path drawn in white on a black background, as shown in centre @fig.dataset_example. This means that the only occupied pixels are those taken up by the path. In this grid, the coordinates of the closest non-zero pixel is found by iterating over the entire input image containing the path. The complexity of this operation will be covered in the following sections. Next, the distance between the current pixel and the closest non-zero pixel is calculated. This distance is then compared to a threshold value to determine its value. If it is further away, the resulting penalty from the loss function should be higher. Different values for the threshold and the exponent of the distance calculation were tested to find the best combination. Lastly, the cold map is saved in a structured folder format for later use in training. Later, the created data is put through augmentation to inflate the size of the dataset and increase its diversity.

==== Finding the distance to the desired path <c4:cold_maps.dist>

The algorithm for finding the distance to the closest point on the desired path is shown in @code.distance_grid. 

#listing([
  ```python
occupied = []
for i in range(binary.shape[0]):
  for j in range(binary.shape[1]):
    if binary[i, j] != 0:
      occupied.append((i, j))

h, w = binary.shape
nearest_coords = np.zeros((h, w, 2), dtype=int)

for i in range(binary.shape[0]):
  for j in range(binary.shape[1]):
    if binary[i, j] == 0:
      min_dist = float('inf')
      nearest_coord = (i, j)
      for x, y in occupied:
        d = hypot(i - x, j - y)
        if d < min_dist:
          min_dist = d
          nearest_coord = (x, y)
      nearest_coords[i, j] = nearest_coord
    else:
      nearest_coords[i, j] = (i, j)
  
  ```
],
caption: [Non-parallelized code for finding the nearest point on the path.]
) <code.distance_grid>

The algorithm in @code.distance_grid starts by creating an array of coordinates based on the `binary` map created with the `threshold` function from the OpenCV library. This `binary` map contains every non-black pixel in the input image, which in this case is the path drawn on a black background. With these occupied pixels stored in an array, the algorithm then iterates over every grid point of the `nearest_coords` grid, created to be the same size as the input image. For every point in the grid, the algorithm checks if the point is on the path. If it is, the algorithm assigns the current point's coordinates to the `nearest_coords` grid. If the point is not on the path, the algorithm iterates over every occupied pixel and calculates the distance between the current point and the occupied pixel. If the distance is less than the current minimum distance, the minimum distance is updated and the coordinates of the closest point are saved. This is repeated for every occupied pixel, and the coordinates of the closest point are saved in the `nearest_coords` grid. This process is repeated for every point in the grid until every point has been assigned the coordinates of the closest point on the path. This grid will later be used under the name `coords`.

The shown algorithm is not parallelized and has a complexity of $cal(O)(n^2)$, where $n$ is the size of the input image. This is due to the nested `for`-loops used in the algorithm. While not a great complexity, it is a vast improvement over its earlier iteration which was $cal(O)(n^4)$#footnote([The original implementation can be seen in `dataset/lib.py:process_rows` in the GitLab repository.]). The actual implementation of this algorithm is parallelized, but the non-parallelized form is shown here. The first iteration of the algorithm took 73 minutes to complete on a $400 times 400$ image, while the parallelized version took 8 minutes on an 8-core CPU. This non-parallelized version takes roughly 30 seconds to complete on the same image, with the parallelized version taking just a few seconds on a full $400 times 400$ image. Further improvements are likely possible to be made both to the complexity of the implementation and parallelization could be distributed to a GPU or the cloud for even faster computation, but this remains future work.

==== Creating the cold map <c4:cold_maps.create>

To start the creation of the cold map, a distance grid is created using Pythagoras' theorem between the coordinates of the point of the grid and the coordinates saved within, retrieved from the aforementioned `coords` variable. A masking grid is then created by comparing the distance grid to a threshold value. This results in each grid point being calculated using:

$
d_(i j) = sqrt((i - c_(i j 0))^2 + (j-c_(i j 1))^2)
$ <eq:distance>

$ d t_(i j) = cases(
  #align(right)[#box[$d_(i j)$]]& "if" d_(i j) < t,
  #align(right)[#box[$t + (d_(i j) - t)^e$]]& "otherwise"
) $

where $c$ = `coords`, $c_(i j 0)$ = `coords[i, j][0]`, $t$ is the threshold value, and $e$ is the exponent value. All three of these can be seen as function parameters in the function declaration in @code.coldmap. The distance grid is then normalized to a range of 0 to 255 to minimize space usage such that it fits within a byte, i.e. an unsigned 8-bit integer. This is done by subtracting the minimum value and dividing by the range of the values. Alternatively, the `normalize` parameter can be set to another value, as usage within a loss function would prefer a value between 0 and 1 (as detailed in @c4:loss). The resulting grid is then saved as a cold map. The resulting cold map can be seen in the rightmost image in @fig.dataset_example.





#listing([
  ```python
def coords_to_coldmap(coords, threshold: float, exponent: float, normalize: int = 1):
  rows, cols = coords.shape[0], coords.shape[1]

  distances = np.zeros((rows, cols), dtype=np.float32)
  for i in range(rows):
    for j in range(cols):
      distances[i, j] = hypot(i - coords[i, j][0], j - coords[i, j][1])
  
  distances_c = distances.copy()
  mask = distances > threshold
  distances_c[mask] = threshold + (distances[mask] - threshold) ** exponent
  
  distances_c_normalized = normalize * (distances_c - distances_c.min()) / (distances_c.max() - distances_c.min())
  
  return_type = np.uint8 if normalize == 255 else np.float32

  return distances_c_normalized.astype(return_type)
  ```
],
caption: [Non-parallelized code for finding the nearest point on the path.]
) <code.coldmap>



To figure out the optimal values for the threshold and exponent, a grid search was performed. The grid search was done by iterating over a range of values for both the threshold and the exponent. The resulting cold maps were then evaluated by a human to determine which combination of values resulted in the most visually appealing cold map. For a $400 times 400$ image, the optimal values were found to be $t=20$ and $e=1.25$. The grid of results can be seen in figure @fig.coldmaps_comp. In testing, the value of $e=1$ was excluded as it had no effect on the gradient produced in the cold map, meaning all values of $t$ produced the same map.

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

While pretty, these coldmaps can be difficult to understand. Therefore, @fig.dataset_3d shows the 3D plots of the generated cold maps with exponent values $e in {0.50, 0.75, 1.25}$. While the 2D plots with $e < 1$ seem to the eye to be the plots that more greatly penalizes larger distances, the 3D plots reveal that while the points that are further away are more penalized, the gradient is not as steep as the 2D plots would suggest. While heavily penalized, the slope does a poor job of pointing the gradient in the right direction. Then if a point is close, it will experience rapid change that forces it closer to the true path. This is not the desired effect of this loss function, as it should be more lenient to points that are close to the true path. Thus, when $e > 1$, the slope is much more gentle closer to the true path, and steeper further away, which is the desired effect. So, despite the opposite being the intuitive point to take away from glancing at the 2D plots, the 3D plots reveal that the exponent value $e$ should be greater than 1, and thus the value of $e=1.25$ was chosen for the cold maps and defined as the default for the function in @code.coldmap.

#std-block(breakable: false)[
  #v(-1em)
  #box(
    fill: theme.sapphire,
    outset: 0em,
    inset: 0em,
  )
  #figure(
  image("../../../figures/img/dataset_example/coldmaps_comp_3d_2.png"),
caption: [3D plots of cold maps along with their 2D counterparts.]
) <fig.dataset_3d>
]


Finally, a comparison between the retrieved satellite image of an intersection, the optimal path through it, and the cold map generated by the process described above are shown in @fig.dataset_example. This highlights the importance of the cold map in the training process as opposed to the single line path. The cold map allows for a more lenient path to be generated, as the model is not penalized for deviating slightly from the path. 


#let fig1 = { image("../../../figures/img/dataset_example/map.png") }
#let fig2 = { image("../../../figures/img/dataset_example/map_path_thin.png") }
#let fig3 = { image("../../../figures/img/dataset_example/cold_map_new.png") }

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
caption: [Example of satellite image, next to the desired path through with. To the far right, the generated cold map is shown with threshold $t=20$ and exponent $e=1.25$. Notice how it is only the points very close to the path that are very cold, while the rest of the map is warmer the further away it is.]
) <fig.dataset_example>
]

=== Data Augmentation <c3:data_augmentation>


=== Dataset Structure <c3:dataset_structure>