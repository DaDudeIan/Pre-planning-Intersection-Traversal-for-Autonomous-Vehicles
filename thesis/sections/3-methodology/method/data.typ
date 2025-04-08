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

#let colouration(t, s) = box(text(t, fill: gradient.linear(angle: 0deg, (fuchsia, 0%), (red, 25%), (blue, 50%), (green, 75%), (yellow, 100%)), size: s))

#let pat = pattern(size: (5pt, 6pt))[
  #place(dx: 1pt, dy: -2pt, line(start: (0%, 100%), end: (100%, 0%), stroke: 4pt + black))
]

#let distortion(t, s) = box(text(t, fill: pat, size: s))

Creating large datasets is a very time consuming tasks, scaling directly with the complexity and workflows structured around its creation. For the dataset created during this project, the workflow was as follows: Find a suitable intersection for the dataset. Copy the coordinates for the center of the intersection. Use the found coordinates in the satellite script described in @c4:sat.impl to download satellite images. Through trial and error, rotate the downloaded satellite image to align entry with bottom of the image.

Once a bunch of satellite images were downloaded, a small python script was used to automatically distribute each intersection image to their own folder and within each folder, create the structure shown in @listing:dataset_structure. #acr("GIMP") was chosen as the software to draw the paths through the intersections. First, another small script distributed a `.xcf` file to each intersection folder. `.xcf` is the file format for #acr("GIMP") projects. This base `.xcf` was defined to be 400x400 pixels and contained a black background and three empty layers named "left", "right", and "ahead". This massively simplified the process of creating the paths by not having to create a new project every time a new intersection was to be processed.

For each of the paths drawn in #acr("GIMP"), they were saved individually as a `.png` file. Yet another small script then used these images of the path to create the corresponding #acr("JSON") files containing the entry and exit coordinates of the path as well as generate the cold map. The cold map generated is stored as a `.npy` files, as the values of the cold map are simply between 0 and 1, meaning it does not make sense to store as a #acr("PNG") as there is not high enough values to be discernible to the human eye. The small scripts mentioned can be seen in #text("Appendix X", fill: red). // sat_to_dataset, coldmap_gen, ee_gen

As described, this is a very time consuming process, and despite many hours of work being put into it, the training dataset only consisted of 112 intersections, some of which have very similar satellite images. To enlarge this dataset dramatically, the dataset underwent augmentation. Augmentation can be done in many ways with different methods. A variety of augmentations were chosen for this dataset, including: colouration, distortion, cropping, and zooming. The reason for choosing these will be discussed in their respective sections.

#colouration("COLOURATION", 14pt) is the augmentation regarding the colours making up the image. In this project, this is achieved by adjusting the saturation and hue of the HSV colour space for an image. HSV stands for Hue, Saturation, and Value. Changing the hue of an image is changing the colour tone, meaning that a red image can be turned into a blue image. Changing the saturation is changing the intensity of the colours in an image, resulting in a more vibrant or dull image. Changing the value is changing the brightness of the image, meaning that a dark image can be turned much brighter and vice versa. HSV is generally more intuitive than the RGB colour space, as it is more closely related to how humans perceive colour. 

Concretely, the colouration augmentation was done by randomly changing the hue and saturation of the image. This was done to help the models focus on structural features rather than specific colour cues, i.e. become better at generalizing. Colour augmentations also help the model become more robust to changes in lighting conditions. This is especially prominent when using satellite images from all kinds of areas. Some satellite images appear to have a very low image saturation, while others are more vibrant and sharp. Therefore, teaching the model to understand these different conditions is crucial. So, these colouration augmentations help make the models more adept at generalizing to different conditions.

The code for the hue and saturation augmentation functions can be seen in @listing:hue_a and @listing:sat_a, respectively. The hue augmentation function randomly changes the hue of the image by a value between the lower and upper bounds. The values are the defaults from the official documentation of the function. The saturation augmentation function randomly changes the saturation of the image by a value between 6, 8, and 10. These values were chosen as they were found to be the most visually appealing and interesting when testing the augmentations. Finally, a greyscale augmentation is also implemented, which is simply a call of the `saturation_aug` function with the value 0. This is done to further enhance the models understanding of the structural features of the image.

#let hue_a = std-block(breakable: false)[
  #box(
    fill: theme.sapphire.lighten(10%),
    outset: 1mm,
    inset: 0em,
    radius: 3pt,
  )[#text(white, size: 12pt, font: "JetBrainsMono NFM")[Hue augmentation]] \
  The hue augmentation was done by randomly changing the hue of the image. The hue was changed by a value between -0.5 and 0.5. The image was then converted to a tensor and the hue was adjusted using the `adjust_hue` function from the `torchvision.transforms.functional` module. The resulting image was then converted back to a #acr("PNG") image for easier handling.
  #listing(line-numbering: none,
    ```python
    def hue_aug(image, 
        lower = -0.5, upper = 0.5):
      v = random.uniform(lower, upper)
      img = T.ToTensor()(image)
      img = F.adjust_hue(img, v)
      
      return T.ToPILImage()(img)
    ```,
    caption: [Hue augmentation function.] 
  ) <listing:hue_a>
]


#let sat_a = std-block(breakable: false)[
  #box(
    fill: theme.sapphire.lighten(10%),
    outset: 1mm,
    inset: 0em,
    radius: 3pt,
  )[#text(white, size: 12pt, font: "JetBrainsMono NFM")[Saturation augmentation]] \
  The saturation augmentation was done by randomly changing the saturation of the image. The saturation was changed by a value between 6, 8, and 10. The image was then converted to a tensor and the saturation was adjusted using the `adjust_saturation` function from the `torchvision.transforms.functional` module. The resulting image was then converted back to a #acr("PNG") image for easier handling.
  #listing(line-numbering: none,
    ```python
    def saturation_aug(image, 
        val = [6, 8, 10]):
      v = random.choice(val)
      img = T.ToTensor()(image)
      img = F.adjust_saturation(img, v)
      
      return T.ToPILImage()(img)
    ```,
    caption: [Saturation augmentation function.]
  ) <listing:sat_a>
]

#grid(
  columns: (1fr, 1fr),
  hue_a, sat_a
)

Examples of the colouration augmentations can be seen in @fig:colouration. Each column shows an intersection and its augmented variations. The top row is the original image, the second row is the greyscale augmented image, the third row is the hue adjusted image, and the fourth row is the saturation adjusted image. The greyscale images highlights the structural features of the image. By adjusting the hue, the dominant tones of the image are altered, resulting in dominant parts like vegetation appears as a variety of colours, such as blue or even purple. Adjusting saturation then makes the colours more vibrant or muted, creating anything from intensely vivid scenes to nearly colourless landscapes as also highlighted by the greyscale image.

Seeing these colour augmentation examples, it is clear to see that a large amount of diversity has been introduced to the dataset. Rather than training on a dataset where the colours might be very similar, hence not capturing the real world, the models are trained on a dataset that encapsulates real world colour variations. Furthermore, this motivates the model to focus on structural features rather than specific colour cues, which is crucial for generalization.

#let fig = { image("../../../figures/img/dataset_example/augmentation/augmented.png") }

#std-block(breakable: false)[
  #v(-1em)
  #box(
    fill: theme.sapphire,
    outset: 0em,
    inset: 0em,
  )
  #figure(
  fig,
caption: [Example of the colouration augmentations. The top row is the original image, second row is the greyscale augmented image, third row is the hue adjusted image, and fourth row is the saturation adjusted image.]
) <fig:colouration>
]

#distortion("DISTORTION", 14pt) is the augmentation regarding the quality of the image. In this project, this is achieved by applying two methods of intentionally deteriorating the image quality. The first method is noise augmentation and the second is blurring the image. These augmentation were chosen, as they represent common downfalls when working with satellite images. Depending on the area where images are taken, the images are often more blurry in smaller towns and rural areas, while they are noticeably sharper and more pristine in larger cities like capitol cities. Thus, by incorporating distortion augmentations into the dataset, it becomes much more diverse and a better representation of the quality of images that the models are expected to work with.

The noise augmentation was chosen as it is a common issue with images in general, not just satellite images. Noise is produced in images through various outside factors, such as atmospheric conditions, sensor limitations, and environmental interferences. By adding this noise augmentation to the images, it helps the model learn to ignore these artifacts and focus on the relevant features of the image. The noise augmentation was done by generating Gaussian noise through the `randn` function from the PyTorch library. Furthermore, this kind of augmentation has been noted to act as a form of regularization on it own @noise, as it prevents the model from overfitting to these clean, ideal, and pristine images that are common for very populated areas.

The blur augmentation was chosen as blur is a common issue with satellite images, particularly when using images from smaller cities, rural areas, older images, or far out of the cities on the roads where population density is significantly lower. Applying a Gaussian blur simulates these conditions by intentionally deteriorating the image quality, making it more representative of real-world scenarios. This helps the model learn to extract structural features from the image, despite the images being obscured. This augmentation is particularly useful because it forces the model to focus on the structural features of the image, rather than the fine details. This is crucial for the model to generalize well to unseen data, as it is not expected to see the same images during inference as it did during training. 

The code for the noise and blur augmentations can be seen in @listing:noise_a and @listing:blur_a, respectively. The noise augmentation function generates random noise using a normal distribution with a mean of 0 and a standard deviation between 0.1 and 0.5. The noise is then added to the image and the resulting image is clamped to a value between 0 and 1 before being returned as an image. The blur augmentation function applies a Gaussian blur to the image using a kernel size of 5, 7, or 9 and a sigma value of 1.5, 2, or 2.5. The kernel size and sigma values were chosen through testing to find the most visually appealing results, i.e. distortions great enough to distort the details of the image, but not so much that the image becomes unrecognizable.

#let hue_a = std-block(breakable: false)[
  #box(
    fill: theme.sapphire.lighten(10%),
    outset: 1mm,
    inset: 0em,
    radius: 3pt,
  )[#text(white, size: 12pt, font: "JetBrainsMono NFM")[Noise augmentation]] \
  The noise augmentation was done by generating random noise and adding it to the image. The noise was generated using a normal distribution with a mean of 0 and a standard deviation between 0.1 and 0.5. The noise was then added to the image and the resulting image was clamped to a value between 0 and 1. The resulting image was then converted back to a #acr("PNG") image for easier handling.
  #listing(line-numbering: none,
    ```python
def noise_aug(image, mean = 0, std_l = 0.1, std_u = 0.5):
  img = T.ToTensor()(image)
  std = random.uniform(std_l, std_u)
  noise = randn(img.size()) * std + mean
  img = img + noise
  img = torch.clamp(img, 0, 1)
  
  return T.ToPILImage()(img)
    ```,
    caption: [Noise augmentation function.] 
  ) <listing:noise_a>
]


#let sat_a = std-block(breakable: false)[
  #box(
    fill: theme.sapphire.lighten(10%),
    outset: 1mm,
    inset: 0em,
    radius: 3pt,
  )[#text(white, size: 12pt, font: "JetBrainsMono NFM")[Blur augmentation]] \
  The blur augmentation was done by applying a Gaussian blur to the image. Through testing, the kernel sizes of 5, 7, and 9 were chosen, as well as the sigma values of 1.5, 2, and 2.5. After randomly selecting the combination of kernel size and sigma, the image is converted to a tensor and the blur is applied. The resulting image is then converted back to a #acr("PNG") image for easier handling.
  #listing(line-numbering: none,
    ```python
def blur_aug(image, 
    kernel_size = [5, 7, 9], 
    sigma = [1.5, 2, 2.5]):
  kernel_size = choice(kernel_size)
  sigma = choice(sigma)
  img = T.ToTensor()(image)
  img = F.gaussian_blur(img, kernel_size, sigma)
  
  return T.ToPILImage()(img)
    ```,
    caption: [Blur augmentation function.]
  ) <listing:blur_a>
]

#grid(
  columns: (1fr, 1fr),
  hue_a, sat_a
)

Examples of the distortion augmentations can be seen in @fig:distortion. The image to the far left-hand side is the original image, the top row is the noise augmented images, and the bottom row is the blur augmented images. The noise augmented images show the image with added noise, making certain features difficult to see and the image more obscured. The blur augmented images show the image with a Gaussian blur applied, making the image more obscured and less sharp. Seeing these examples, it is clear to see that, again, a large amount of diversity has been introduced to the dataset. The introduction of noise and blur further help the model generalize better by focusing on structural features rather than specific details. This addition to the dataset broadens the ability of the models and teaches them to perform better on suboptimal images.


#let sat_image = { image("../../../figures/img/dataset_example/augmentation/sat_image.png") }
#let noise_aug = { image("../../../figures/img/dataset_example/augmentation/noise_aug.png") }
#let blur_aug = { image("../../../figures/img/dataset_example/augmentation/blur_aug.png") }

#std-block(breakable: false)[
  #v(-1em)
  #box(
    fill: theme.sapphire,
    outset: 0em,
    inset: 0em,
  )
  #figure(
  grid(
    columns: (1fr, 3fr),
    column-gutter: 1mm,
    align: (horizon, center),
    grid.cell(rowspan: 2, sat_image),
    noise_aug, blur_aug
  ),
caption: [Example of the distortion augmentations. The far left image is the original image, the top row is the noise augmented images, and the bottom row is the blur augmented images.]
) <fig:distortion>
]



#text("CROP AND ZOOM", 14pt) is the last augmentation adopted to expand the dataset. They are very common spatial augmentations used on image datasets. They are typically selected to make the models trained more robust against non-centred images and to make the models more adept at generalizing to different scales. If centring is not an issue, then cropping is still a common method for classification tasks, as it helps the model focus on the relevant features of the image. Zooming is also a common augmentation, as it helps the model learn different spatial scales of whatever it is being taught to handle. Also, as discussed earlier in @c4:sat, the zoom level of images in the dataset is set to a value of 18, but these images still appear to consist of vastly different sized intersection, thanks to the fact that some intersection are simply larger than others. Thus, using a zoom augmentation, helps the models generalize better to different scales of intersections. Furthermore, this should also help the model gain a better understanding of road width and not overfit to any particular width present in the dataset.

So, both of these augmentations help the trained models become scale invariant, which is a desired trait of models set to perform the task at hand, as the scale and size of intersection images gotten through satellite images can vary greatly. The crop augmentation was done by randomly cropping the satellite image and its corresponding paths. The crop size is determined by a factor of the original image size, with a default value of 0.8. The cropped image and paths are then resized back to the original dimensions to maintain consistency. This augmentation helps the model focus on different parts of the image and improves its robustness to non-centred features, as is highly relevant for this project. The zoom augmentation was done by resizing the image to a larger size and then cropping it back to the original size. Through testing, the zoom factors of 1.4 to 1.9 were chosen. After randomly selecting a zoom factor, the image is resized and cropped to simulate zooming in.


#let hue_a = std-block(breakable: true)[
  #box(
    fill: theme.sapphire.lighten(10%),
    outset: 1mm,
    inset: 0em,
    radius: 3pt,
  )[#text(white, size: 12pt, font: "JetBrainsMono NFM")[Crop augmentation]] \
  The crop augmentation was done by randomly cropping the satellite image and its corresponding paths. The crop size is determined by a factor of the original image size, with a default value of 0.8. The cropped image and paths are then resized back to the original dimensions to maintain consistency. This augmentation helps the model focus on different parts of the image and improves its robustness to non-centred features.
  #listing(line-numbering: none,
    ```python
def crop_aug(sat_image, paths, factor = 0.8):
    h, w = sat_image.size
    new_h = int(h*factor)
    new_w = int(w*factor)
    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)
    img = F.crop(sat_image, top, left, 
                 new_h, new_w)
    scaled_img = F.resize(img, (h, w))
    
    path_images = []
    for p in paths:
        path = p["path_line"]
        path_img = F.crop(path, top, left, 
                          new_h, new_w)
        scaled_path = F.resize(path_img, 
            (h, w), interpolation=LANCZOS)
        path_images.append(scaled_path)
    
    return scaled_img, path_images
    ```,
    caption: [Crop augmentation function.] 
  ) <listing:crop_a>
]


#let sat_a = std-block(breakable: true)[
  #box(
    fill: theme.sapphire.lighten(10%),
    outset: 1mm,
    inset: 0em,
    radius: 3pt,
  )[#text(white, size: 12pt, font: "JetBrainsMono NFM")[Zoom augmentation]] \
  The zoom augmentation was done by resizing the image to a larger size and then cropping it back to the original size. Through testing, the zoom factors of 1.4 to 1.9 were chosen. After randomly selecting a zoom factor, the image is resized and cropped to simulate zooming in. The resulting image is then converted back to a #acr("PNG") image for easier handling.
  #listing(line-numbering: none,
    ```python
def zoom_aug(image, paths, 
             zoom_range = (1.4, 1.9)):
  zoom_factor = uniform(*zoom_range)
  tmp_h = int(image.size[0]*zoom_factor)
  tmp_w = int(image.size[1]*zoom_factor)
  img = F.resize(image, (tmp_h, tmp_w)) 
  img = F.center_crop(img, image.size)
  
  path_images = []
  for p in paths:
    path = p["path_line"]
    path_img = F.resize(path, 
                        (tmp_h, tmp_w))
    path_img = F.center_crop(path_img, 
                             image.size)
    path_images.append(path_img)
  
  return img, path_images
    ```,
    caption: [Zoom augmentation function.]
  ) <listing:zoom_a>
]

#grid(
  columns: (1fr, 1fr),
  hue_a, sat_a
)

Examples of the crop and zoom augmentations can be seen in @fig:crop_zoom. The far left image is the original image, the top row is the crop augmented images, and the bottom row is the zoom augmented images. The crop augmented images show the image cropped to a different part of the image, with the associated paths being cropped by the same factors. This was a necessary step, separating these augmentations from the previous, as the paths through the image were not the exact same as when the image underwent colouration or distortion augmentations. The same is true for the zoom augmented image. Here, the paths also needed to undergo the same augmentation as the satellite image. Once again, these augmentation have clearly impacted the diversity of the dataset, as the models are now trained on images that are not centred and images that are zoomed in, which is important for generalization.

#let sat_image = { image("../../../figures/img/dataset_example/augmentation/sat_image.png") }
#let crop_zoom_aug = { image("../../../figures/img/dataset_example/augmentation/crop_zoom_aug.png") }

#std-block(breakable: false)[
  #v(-1em)
  #box(
    fill: theme.sapphire,
    outset: 0em,
    inset: 0em,
  )
  #figure(
  grid(
    columns: (1fr, 3fr),
    column-gutter: 1mm,
    align: (horizon, center),
    sat_image, crop_zoom_aug
  ),
caption: [Example of the crop and zoom augmentations. The far left image is the original image, the top row is the crop augmented images, and the bottom row is the zoom augmented images.]
) <fig:crop_zoom>
]

All in all, these augmentations were selected to expand the dataset by introducing real-world inspired variations of the satellite images. These augmentation mimic the unpredictable conditions posed by using satellite imagery. By employing techniques such as hue and saturation adjustments, greyscale conversions, noise injection, blur filtering, and spatial transformations like cropping and zooming, the augmented dataset now contains diverse lighting, quality, and scale entries. This strategy of employing such a variety of augmentations not only ballooned the size of the dataset immensely, but also made the dataset more representative of the real-world conditions the models are expected to work with, meaning that it should be more robust and better at generalizing to unseen data.

Other augmentation techniques were considered for this project. Rotation, for example, was considered to further increase the diversity of the dataset by introducing variations in orientation, which could help the model learn to recognize paths from different angles. However, early on in this project, it was decided that the entry for each path should be at the bottom of the satellite image. Thus, rotation the images could have undesired consequences for the models' ability to generalize with this constraint. Flipping was also considered, but was ultimately left out, as it was deemed to have too little of a potential impact since many intersections already go in all directions. Translation was also considered, but was left out, as it would require the paths to be redrawn as they would no longer reach the edges of the image, which was a factor this entire set out to combat. Finally, a very common augmentation used in segmentation, is the act of using cutmix and its constituent parts, namely cutout and mixup. These were, however, also left out, as it is assumed that the satellite images do not contain holes or other artefacts that would be introduced by these augmentations. Furthermore, cutmix might disrupt the spatial features understood by the model. These may be introduced in future work, as they may ultimately increase the robustness of the models in environments that have blocking features.


=== Dataset Structure <c3:dataset_structure>

#text("UPDATE TO INCLUDE CLASS_LABELS", fill: red, weight: "black")

To maintain an ease-of-use principle for this project, the dataset was structured in a way that allows for easy loading of the data. This includes building the dataset in a logical way, and creating a class that can load the dataset gracefully. This is especially important as the paths in a dataset can vary in number, so custom loading is necessary. Thus, the dataset is structured like shown in the listing below

#listing(line-numbering: none, [
  ```text
dataset/
├── train/
│   ├── intersection_001/
│   │   ├── satellite.png
│   │   ├── class_labels.npy
│   │   ├── paths/
│   │   │   ├── path_1/
│   │   │   │   ├── path_line.png
│   │   │   │   ├── path_line_ee.json
│   │   │   │   ├── cold_map.npy
│   │   │   ├── path_2/
│   │   │   │   ├── path_line.png
│   │   │   │   ├── path_line_ee.json
│   │   │   │   ├── cold_map.npy
│   │   │   ├── path_3/
│   │   │   │   ├── path_line.png
│   │   │   │   ├── path_line_ee.json
│   │   │   │   ├── cold_map.npy
├── test/
│   ├── intersection_001/
│   │ ...

  ```
],
  caption: [Folder structure of the dataset. Each `intersection_XXX` folder contains a satellite image of an intersection and a `paths` folder containing the paths through the intersection. Each path folder contains the path line, the path's entry and exit points, and the cold map for the path in a `.npy` format.],
) <listing:dataset_structure>

Firstly, the dataset is split into two separate parts, `train` and `test`. This is done to ensure that the model is not overfitting to the training data. This is achieved by training the model on the `train` dataset and testing/validating it on the `test` dataset. To ensure that the models generalize well to the task as hand, the `test` dataset should contain intersections that are completely absent from the `train` dataset. This is done to ensure it does not fall into the simple trap of memorizing the training data and created really good results that can be considered false positives as it supposedly has never seen the data before. 

This `train`/`test` split in the dataset is created in the folder structure instead of using the simpler functionalities offered by PyTorch. PyTorch offers a `random_split` function from its utility sub-library. This function takes in some dataset declared as a PyTorch `Dataset` object, as shown below in @c3:dataset_structure:dataset_class, and splits it based on a given ratio. This is a simple way to split the dataset, but, as is the case of the created dataset, some images are very similar, meaning that the split does not achieve the desired effect and the model overfits to the training data. Thus, a completely different set of intersections is used for the `test` dataset.

Each `intersection_XXX` folder contains a satellite image saved as a #acr("PNG"). Accompanying this image, is the `paths` folder, which contains a folder for each path through the intersection. Each path folder contains the path line image, currently saved as a #acr("PNG") as well, a #acr("JSON") file containing the entry and exit points of the path in relation to the image, not the global coordinates, and the corresponding cold map saved as a `.npy` file.

==== Dataset class <c3:dataset_structure:dataset_class>

#text("UPDATE ALL TO REFLECT LATEST CHANGES. MAYBE ALSO DIFFERENT TYPES.", fill: red, weight: "semibold")

#let init = {text("__init__", font: "JetBrainsMono NFM")}
#let len = {text("__len__", font: "JetBrainsMono NFM")}
#let getitem = {text("__getitem__", font: "JetBrainsMono NFM")}

To be able to easily load a satellite image and its corresponding paths, entry/exit points, and cold maps, a `IntersectionDataset` class was created, built on top of the PyTorch `Dataset` class. To implement this class, three functions must be created, namely `__init__`, `__len__`, and `__getitem__`. 

#std-block(breakable: true)[
  #box(
    fill: theme.sapphire.lighten(10%),
    outset: 1mm,
    inset: 0em,
    radius: 3pt,
  )[#text("__init__", white, size: 12pt, font: "JetBrainsMono NFM")] #h(0.35em) is the function called when the class is instantiated. It initializes the dataset with the root directory of the dataset, a transform function, and a path transform function. The root directory is the directory where the dataset is stored, the transform function is a function that can be applied to the satellite image, and the path transform function is a function that can be applied to the path line. The root directory passed to the instantiation should that of either the training or test dataset within the dataset root folder. The transforms are simply `ToTensor` functions provided by PyTorch. The `__init__` function also creates a list of all the paths in the dataset by listing all directories found in the `paths` folders. The code for the `__init__` function can be seen in @listing:dataset_structure_init below.
  \ #v(-1cm) \
    #listing([
    ```python
def __init__(self, root_dir, transform = None, path_transform = None):
  self.root_dir = root_dir
  self.transform = transform
  self.path_transform = path_transform
  
  self.path_dirs = glob.glob(f'{root_dir}/*/paths/*')
    ```
  ],
    caption: [Code snippet of the #init function for the dataset.],
  ) <listing:dataset_structure_init>
  Here the library `glob` is used to list all directories found in the `paths` folders. It is a useful library that allows for the use of wildcards in the path, making it easy to find all directories in the `paths` folders. This approach was used as it was not certain that all `paths` folders contained three subfolders, meaning that there might be inconsistencies in how the data is structured across different intersections. This approach ensures that all paths are found, regardless of the structure of the `paths` folder.
]

#std-block(breakable: true)[
  #box(
    fill: theme.sapphire.lighten(10%),
    outset: 1mm,
    inset: 0em,
    radius: 3pt,
  )[#text("__len__", white, size: 12pt, font: "JetBrainsMono NFM")] #h(0.35em) is another function required by the PyTorch `Dataset` class. It returns the length of the dataset. Thanks to the initialization of the dataset in the `__init__` function, the length of the dataset is simply the number of paths in the dataset. The code for the `__len__` function can be seen in @listing:dataset_structure_len below.
  \ #v(-1cm) \
    #listing([
    ```python
def __len__(self):
  return len(self.path_dirs)
    ```
  ],
    caption: [Code snippet of the #len function for the dataset.],
  ) <listing:dataset_structure_len>
]

#std-block(breakable: true)[
  #box(
    fill: theme.sapphire.lighten(10%),
    outset: 1mm,
    inset: 0em,
    radius: 3pt,
  )[#text("__getitem__", white, size: 12pt, font: "JetBrainsMono NFM")] #h(0.35em) is one of the most crucial functions of the dataset class. The signature of the function is simply `__getitem__(self, idx)`, where `idx` is the index of the path to be loaded. First, the function retrieves the directory of the path at the given index. Then, it loads the satellite image from the intersection directory and applies the transform function to it. This is achieved by moving up by two directories, i.e. getting the satellite image from the `intersection_XXX` folder. It then loads the path itself and applies the path transform function to it. These transforms are simply `ToTensor` as provided by PyTorch. Then, for the path being indexed, the function loads the #acr("JSON") file containing the entry and exit points of the path, and the cold map. The entry/exit data is loaded from a #acr("JSON") file, and the cold map is loaded from a `.npy` file. All of this data is then stored in a dictionary and returned as the sample. The code for the `__getitem__` function can be seen in @listing:dataset_structure_getitem below.
  \ #v(-1cm) \
    #listing([
    ```python
def __getitem__(self, idx):
  path_dir = self.path_dirs[idx]
  
  # Load satellite image (../../satellite.png)
  satellite_path = os.path.join(os.path.dirname(os.path.dirname(path_dir)), 'satellite.png')
  satellite_img = Image.open(satellite_path).convert('RGB')
  
  if self.transform:
    satellite_img = self.transform(satellite_img)
      
  # load path line image (./path_line.png)
  path_line_path = os.path.join(path_dir, 'path_line.png')
  path_line_img = Image.open(path_line_path).convert('L')
  
  if self.path_transform:
    path_line_img = self.path_transform(path_line_img)[0]
      
  # load E/E json file (./path_line_ee.json)
  json_path = os.path.join(path_dir, 'path_line_ee.json')
  with open(json_path) as f:
    ee_data = json.load(f)
      
  # load cold map npy (./cold_map.npy)
  cold_map_path = os.path.join(path_dir, 'cold_map.npy')
  cold_map = np.load(cold_map_path)
  
  # return sample
  sample = {
    'satellite': satellite_img,
    'path_line': path_line_img,
    'ee_data': ee_data,
    'cold_map': cold_map
  }
  return sample
    ```
  ],
    caption: [Code snippet of the #getitem function for the dataset.],
  ) <listing:dataset_structure_getitem>
]

The dataset is then simply instantiated as such:
#listing([
  ```python
dataset = IntersectionDataset(root_dir=dataset_dir,
                              transform=ToTensor(),
                              path_transform=ToTensor()) 
  ```
], caption: [Instantiation of the dataset.]) <listing:dataset_structure_instantiate>

where `dataset_dir` is the path to either the training or test dataset folders. Creating the dataloader is as simple as:
#listing([
  ```python
dataloader = DataLoader(dataset, 
                        batch_size=b, 
                        shuffle=True, 
                        num_workers=num_workers)
  ```
], caption: [Creating a dataloader for the dataset.])

Arguments passed to the `DataLoader` initializer are the dataset from @listing:dataset_structure_instantiate, the batch size, whether the dataset should be shuffled, the number of workers to use for loading the data, and the ability to give it a custom collate function, which is not necessary in this case as the default function `default_collate` handles the data gracefully. `num_workers` is found using the `multiprocessing` library as it easily finds the number of available computation cores. An example of the dataloader in action is shown in @fig.dataloader_example, where an example batch from the `DataLoader` is shown. The batch contains four paths, showcasing the path and the associated satellite image.

#let fig1 = { image("../../../figures/img/dataset_example/loader_2.png") }

#std-block(breakable: false)[
  #v(-1em)
  #box(
    fill: theme.sapphire,
    outset: 0em,
    inset: 0em,
  )
  #figure( fig1,
  caption: [Example batch from the `DataLoader` with `batch_size = 4`.]
) <fig.dataloader_example>
]