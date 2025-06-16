# Pre-planning Intersection Traversal for Autonomous Vehicles

## Overview

This project focuses on the research and development of methodologies for pre-planning the traversal of complex urban intersections by autonomous vehicles. It involves dataset creation, deep learning model development, and evaluation, likely culminating in the findings presented in the associated thesis. The aim is to enhance the safety and efficiency of autonomous navigation in challenging intersection scenarios.

## Context

This repository is part of my Master's thesis project, which explores the intersection of autonomous vehicle navigation and deep learning. The research emphasizes pre-planning strategies that can be applied before an autonomous vehicle approaches an intersection, leveraging advanced machine learning techniques to predict optimal paths and ensure safe traversal.

## Key Features

* **Intersection-Specific Pre-planning:** Focus on strategies developed before an autonomous vehicle enters an intersection.
* **Deep Learning Models:** Exploration and implementation of various architectures such as UNet, DeepLabV3, Swin Transformers, and Vision Transformers (ViT) for tasks like semantic segmentation or path prediction.
* **Custom Dataset Pipeline:** Includes tools for data collection (possibly from satellite imagery), augmentation, and processing tailored for intersection analysis.
* **Comprehensive Research:** The project is well-documented through a dedicated `thesis/` directory, indicating a thorough academic investigation.

## Workspace Structure

The project is organized into the following main directories:

* `/` (Root Directory):
  * `README.md`: This file.
  * `requirements.txt`: A list of Python dependencies required to set up the project environment. Simply generated using `pip freeze > requirements.txt`.
  * `LICENSE`: Contains the licensing information for this project.
  * `local_modules.py`: Likely contains custom Python utility functions or classes used across the project.
  * `Contract.pdf`: An administrative document related to the project.

* `dataset/`: Manages all aspects of the dataset used for training and evaluating models.
  * **Notebooks** (`augmentation.ipynb`, `coldmap_gen.ipynb`, `dataloader.ipynb`, `dataset_creator.ipynb`, `path_entry_exit.ipynb`, `paths_combiner.ipynb`, `sat_to_dataset.ipynb`, `testing.ipynb`): Jupyter notebooks for various data processing tasks including data augmentation, colormap generation, data loading, dataset creation from raw sources (e.g., satellite images), path analysis, and dataset integrity checks.
  * **Libraries** (`dataset_lib.py`, `IntersectionDataset.py`): Custom Python scripts for dataset management and defining the `IntersectionDataset` class.
  * **Data Storage**:
    * `dataset/train/` & `dataset/test/`: Directories holding the training and testing datasets, respectively.
    * `unprocessed/`: Contains raw data (e.g., map images) before processing.
    * `img/`: Stores images related to the dataset, such as augmented samples, visualizations from the dataloader, or satellite snippets.

* `loss/`: Contains code, experiments, and libraries related to the loss functions used for training the models.
  * `loss_lib.py`, `topo_lib.py`: Custom Python libraries for defining and implementing various loss functions, potentially including topological or specialized geometric losses.
  * `loss_testing.ipynb`: A Jupyter notebook for experimenting with and evaluating different loss functions.
  * **Visualizations & Data**: Includes various `.png` images (e.g., `cmap_loss_comparison.png`) and `.npy` files, which are likely plots comparing loss performance or saved numerical data from experiments.

* `models/`: Houses the machine learning models, training scripts, and saved results.
  * **Architectures** (`unet/`, `deeplabv3/`, `swin/`, `vit/`): Subdirectories for different deep learning model architectures explored in the project.
  * `training/`: Likely contains scripts or notebooks for training the models.
  * `results/`: Stores the outputs, trained weights, and performance metrics of the models.

* `sat/`: This directory appears to be dedicated to satellite imagery processing or a specific component/model abbreviated as "SAT".
  * **Notebooks** (`sat.ipynb`, `act_funcs.ipynb`, `augs.ipynb`): Jupyter notebooks for experiments related to satellite image handling, testing activation functions, or specific augmentations in this context.
  * **Images & Data**: Contains images like `map.png`, `mapX.png`, and potentially other data files relevant to this module.

* `thesis/`: Contains all files and materials related to the academic thesis that documents this research project.
  * `main.typ`, `main.pdf`: The source file (Typst format) and the compiled PDF of the thesis document.
  * `references.bib`, `references.yaml`: Bibliography and reference management files.
  * `sections/`: Subdirectory likely containing individual chapters or sections of the thesis in `.typ` format.
  * `figures/`, `img/`: Directories storing images, diagrams, and plots used in the thesis.
  * `template.typ`, `acronyms.yaml`, etc.: Supporting files for the thesis compilation and formatting.

## Check Locally

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Pre-planning-Intersection-Traversal-for-Autonomous-Vehicles
    ```
    Note: `requirements.txt` is created with `pip freeze > requirements.txt`, so contents may vary based on your local environment.

## License

This project is distributed under the terms specified in the `LICENSE` file. Please review it for more details on permissions and limitations.

