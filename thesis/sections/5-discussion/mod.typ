#import "../../lib/mod.typ": *

// + How can pixel-subset-based deep learning approaches be optimized to improve accuracy and efficiency in path planning for autonomous vehicles at intersections? How do convolution-based and transformer-based models compare in this context?
// + Is it possible to design a loss function that effectively captures the similarity between generated and desired paths for autonomous vehicles without forcing exact matches?
// + Is it possible to create a dataset that allows for the training of a model, such that the data is not too stringent to a singular path?

= Discussion #checked <c6:Discussion>

This section presents the discussion of the results, methods, and broader implications of the work carried out in this thesis. It begins with the integration of the proposed system into real-world infrastructure, including its compatibility with V2X communication, considerations related to ISO 26262 compliance, and how it should actually work with current systems active in vehicles. Following this, the notable shortcomings of the project are outlined, including performance limitations, data dependencies, and the limited improvements observed with topological loss functions. After this, several technical insights are discussed, such as the behaviour of transformer-based models, the relationship between loss and accuracy, and the outcomes of extended training runs. The next section covers implementation-related considerations, such as the trade-offs between cloud and onboard inference, inference times, and post-processing strategies. Then the potential domain transfer and industrial relevance of this technology will be discussed. Generalization and robustness are then explored, with a focus on seasonal changes and the inductive biases of different model architectures. Finally, the chapter concludes with a discussion of the societal and ethical aspects associated with deploying such a system, including privacy, legality, and the potential impact on traffic and the environment.


== Integration with existing systems #checked <c6:integration>

The system proposed in this thesis is designed to be an integrated component of a vehicle's navigation and path planning system, particularly when operating in a self-driving mode. It is intended to work alongside existing vehicles systems, such as V2X communication and other autonomous driving technologies, to enhance the vehicle's ability to navigate complex intersections safely and efficiently. The fact that this system works purely through the use of machine learning models, means that it should not be the main source of driving instructions for AVs, but rather as a supplementary tool that works where other V2X systems are not available.

V2X has great potential to enhance the effectiveness of AVs by providing real-time information about traffic conditions, road hazards, and other vehicles' intentions. Furthermore, expanding it to handle the task of AIM systems, means that AVs can rely on it even further. However, as discussed, there are severe limitations tied to the use of V2X, such as the need for a robust and reliable communication infrastructure, which may not always be available in all environments. Therefore, the system proposed is meant to be a fallback solution that can operate independently of V2X communication when necessary. With further adjustments, it could even be integrated into existing V2X systems, allowing for a more seamless and efficient operation in environments where V2X is available.

Having all these systems working together and acting as fallbacks for each other, means that the vehicle can operate more safely and efficiently in a wider range of environments. This aspect of having fallback systems is an important part of the safety, as described in ISO 26262. The standard mandates the definition of safe states and requires fault-tolerance behaviour, which in turn requires fallback strategies to ensure safety when an error occurs. The proposed system can be designed to comply with these requirements by implementing a hand-over strategy that allows the vehicle to switch between different systems based on their availability and reliability. That is to say, if the proposed system generates a path in low confidence, then the vehicle should switch to a different system or give control back to the driver. 

The optimal conditions for this system to operate are when vehicles are using an on-board GPS system, both when using it for navigational directions and when it passively shows the vehicle's location on the infotainment system. With this knowledge, the on-board system can check what is ahead of the vehicle and thus perform the path generation only on the potential intersections ahead. In the case of using GPS for navigation, the coordinates of the intersection as found by the GPS system can be passed to the proposed system to fetch the latest satellite imagery and generate a path. This can be done for the entirety of a route, as the GPS system knows the coordinates of all intersections along it.

A likely extension of the system developed in this project, is the ability to handle images that have an arbitrary rotation angle. In the current state, however, the entry to the intersection is assumed to be at the bottom of the satellite image, thus requiring the images to be rotated accordingly. This is hypothesized to work as follows, but is left as future work: When the system is to generate the path for some intersection, the GPS coordinates of the centre of the intersection are given to it, as well as the coordinates of the previous dot along the route, this can be just before the intersection or further back, depending on the resolution of the GPS system. The system can then calculate the angle between the two points, which is the angle that the image should be rotated to align the intersection with the bottom of the image. If the resolution is too poor for this to work, then the system can use the GPS coordinates of the vehicle to determine its position relative to the intersection and thus calculate the angle. 

The final consideration for the integration of this system into existing vehicles is the memory footprint of the models used. It is widely known that EV computers are optimized for inference @comp_power, and can therefore handle large models without issues. Therefore, as shown in @tab:memory-footprint, the models used in this project are not a concern for on-board memory. The table shows the memory footprint in three different categories: the total static model size, which includes the parameters and buffers of the model, the file size of the model, and the maximum memory allocated during inference. The values show that all models have a total static size below 350 MB, with the largest being Swin at 344.34 MB. The maximum memory allocated during inference is also below 3000 MB for all models, with the largest being ViT at 2662.88 MB. This means that the models can be run on most modern EV computers without issues. For training, however, the footprint is significantly larger as the dataloaders and every peripheral used during training also takes up memory, leading to the limited sized models and batch sizes used in this project. Furthermore, a discussion of on-board vs cloud computing will be presented later.

#let tab = [
  #figure(
    {
      tablec(
        columns: 4,
        alignment: (x, y) => (left, center, center, center).at(x),
        header: table.header(
          [Model], [Static Size (MB)], [File Size (MB)], [Max Allocated Memory (MB)]
        ),
        [DeepLabV3+], [44.83], [44.99], [484.36],
        [U-Net], [51.13], [51.17], [1232.27],
        [ViT], [328.57], [328.62], [2662.88],
        [Swin], [344.34], [344.39], [2796.26],
        []
      )
    },
    caption: [Memory footprint of the models used in this project. The values are based on the total static model size, file size, and maximum memory allocated during inference.]
  ) <tab:memory-footprint>
]

#tab

// - Integration with V2X: rely on V2X communication when available; otherwise, fall back to the onboard model. #checked 
// - Memory footprint: on-board memory is not a concern, as models are designed for inference and are lightweight.
// - GPS limitations: GPS alone does not provide sufficient information for precise path planning. #checked
// - Rotation estimation from GPS: determine vehicle orientation using the vector between the current position and the intersection center. #checked
// - Compliance with ISO 26262: implement a hand-over strategy that aligns with functional safety standards. #checked

// rely on v2x when available, otherwise this method
// On-board memory footprint of each model (hardly a problem as models do not require training. Computers optimized for inference.)
// Deeplab: 11,723,541 (Total Static Model Size (Parameters + Buffers): 44.83 MB, Max memory allocated: 484.36 MB, Model file size: 44.99 MB)
// U-Net: 13,395,589 (Total Static Model Size (Parameters + Buffers): 51.13 MB, Max memory allocated: 1232.27 MB, Model file size: 51.17 MB)
// Vit: 86,131,973 (Total Static Model Size (Parameters + Buffers): 328.57 MB, Max memory allocated: 2662.88 MB, Model file size: 328.62 MB)
// Swin: 89,102,013 (Total Static Model Size (Parameters + Buffers): 344.34 MB, Max memory allocated: 2796.26 MB, Model file size: 340.01 MB)

// GPS not enough
// How to find the angle to rotate? When driving with a GPS, use coordinates to the centre. Find angle by finding the angle between the two points. 
// 
// Hand-over strategy in compliance with IDO 26262

== Project Limitations and Challenges #checked <c6:shortcomings>

As it has been pointed out extensively throughout this thesis, there are several limitations and challenges encountered in this project. Chief amongst these is the limitation of the hardware on which these models were trained and tested. The greatest limiting factor was the GPU memory, which restricted the batch size and model size that could be used. This limitation is particularly relevant for transformer-based models, which tend to require more memory than convolutional models and large batch sizes to achieve optimal performance. This is shown in @tab:memory-footprint, where both transformer-based models require more than twice the memory of the convolutional models. 

Some of the other limitations pertaining to this project come with insight into the proposed research questions. Firstly, #RQ(2), which asks whether it is possible to design a loss function that effectively captures the similarity between generated and desired paths without forcing exact matches, has shown that while topology-based losses can provide some benefits, they do not consistently outperform traditional losses like Cross-Entropy. For starters, the continuity loss, which was expected to improve the model's ability to generate connected paths, came from a published research article showing promising results in a different context. However, in this project, it did not yield the expected improvements, especially with the transformer-based models. It did manage to provide cleaner outputs from the convolutional models, but the overall performance did not improve significantly, especially in terms of mIoU. 

The cold map loss function, which was designed to encourage the model to generate paths that are similar to the ground truth paths, also did not yield significant improvements. This is despite the fact that it very closely falls in line with #RQ(2). To justify this claim, the generated cold maps gradual increase in penalty allows for some leeway in the generated paths, therefore not being extremely strict to the exactness of the output, but moreso the structure with respect to the structure of the road. This trait exactly describes the goal of #RQ(2), where the continuity loss does less to answer the question, as it does not focus on the structure of the path with respect to the road, but merely the continuity of the path itself. All-in-all, this suggests that the topology-based losses may not be universally applicable or effective across different model architectures and datasets.

Thus, a key conclusion can be made regarding the effectiveness of topology-based loss functions. While both of the topological functions did demonstrate the capability to influence the generated paths' structure, such as generating cleaner results by visually removing artifacts, these improvements did not translate to a superior quantitative performance, as highlighted by the mIoU of @tab:all_miou. This suggests that for the task tackled in this project, applying these topological losses directly to models yield limited results, especially when compared to well-tested losses like Cross-Entropy. Therefore, their formulations and methodology may need to be refined or adapted to better suit the specific requirements of path planning in autonomous vehicles.

Secondly, #RQ(3) asks whether it is possible to create a dataset that allows for the training of a model, such that the data is not too stringent to a singular path. By "too stringent to a singular path," it is meant that the paths in the dataset should not only be 1 pixel wide, as was the case in this project's infancy. This 1 pixel path ground truth would've proven even more difficult for more models to learn as the class imbalance would be even more pronounced and it would allow for even less generalization. This is due to the fact that, if the ground truths were 1 pixel wide, the models would have to learn unnecessarily precise paths, which could be hugely effected by how precisely centred the intersection is in the image, as well as the resolution of the satellite imagery. Thus, it is claimed that the generated dataset does justify the question posed in #RQ(3), as the width given to the ground truth labels allows for more leniency when the models are being trained. This change did, however, prove to be an insufficient solution to overfitting.

Furthermore, the fact that the models have an extremely heavy tendency to overfit to the training data, is a clear indication that the dataset is not diverse or extensive enough. This is shown in the results chapter, where the mIoU never gets above 0.45, which is a clear indication that the models are not able to generalize well to unseen data. As noted in the section itself, a small value for the mIoU is expected, as the paths are still relatively narrow and minor fluctuations in the path may lead to a lower per-class IoU, effecting the mIoU. A higher mIoU is likely still achievable with more diverse and extensive training data, which would allow the models to learn more generalized features and patterns in the data. 

The imbalanced nature of the dataset is, in this project, tackled by the use of Cross-Entropy loss and, by extension, Binary Cross-Entropy loss. These losses were chosen for their well-known performance throughout literature, and their ability to handle class imbalance to some extent. As shown in this project, CE loss does provide the best results of the loss methods tested. However, it is acknowledged that other loss functions, such as Focal Loss and Dice Loss, could potentially provide better performance in this context. Focal Loss is designed to address class imbalance by down-weighting the loss contribution from well-classified examples, while Dice Loss is particularly effective for imbalanced datasets as it focuses on the overlap between predicted and ground truth regions. 

Finally, to note on the potential shortcomings of the proposed system, it is important to highlight the dependency on fresh satellite imagery. The system relies on up-to-date satellite images to remain reliable, since the predicted paths will be useless if the underlying satellite imagery is outdated or does not reflect the current state of the environment. This is particularly relevant in dynamic environments such as construction zones, where road markings and paths may change frequently. Generally, services like Google Maps, as used in this project, provide updates to their satellite imagery in the range of 1--5 years, depending on the kind of area being mapped. This means that the system may not be able to provide reliable paths in rural areas, as they are updated less frequently than urban areas.

The results achieved in this project do well to showcase the developed method's strengths and weaknesses, as well as the limitations of the dataset and the models used. Despite this, this section has shown that the work done still provides insight into two of the posed research questions, while also highlighting the need for further work in the area of loss function design and dataset diversity. The next section will look deeper into the technical observations and some training insights gained. 

// - Dependency on fresh satellite imagery: the system requires up-to-date imagery to remain reliable, especially in dynamic environments such as construction zones.
// - Visual misclassification: include examples of predictions that appear correct but fail under safety-critical scrutiny.
// - #RQ(2): Evaluates the performance of the topology-based loss functions (e.g., continuity loss), highlighting their limited effectiveness and raising important considerations about how to better capture path similarity without pixel-level matching.
//   - Limited gains from topology-aware losses: the expected benefits of continuity and branching losses were not consistently observed.
// - #RQ(3): Discusses class imbalance and the need for more varied data, which relates to avoiding overfitting to a single type of path and supporting generalization.
//   - Performance plateau: the overall mIoU stagnated below 0.45, indicating a need for more diverse and extensive training data.
//   - Class imbalance: justify the use of Cross-Entropy and Binary Cross-Entropy; reflect on the potential benefits of Focal Loss and Dice Loss.


// Notable shortcomings of project: requires latest satellite images to be completely useful. (like construction zones, etc)

// mIoU never getting above 0.45, pointing out need for more data.

// illustrate the “looks good but fails safety check” issue noted in the results discussion. (Include zoom in on different images.)

// Lacking improvements introduced by topological loss function. 

// Class imbalance. Justify use of CE and BCE. Discuss focal and Dice loss.


== Technical Observations and Training Insights #checked <c6:technical-insights>

An immediate and interesting observation can be made from the mIoU table in @tab:all_miou: the greatest performing model is DeepLabV3+ with just Cross-Entropy loss and at 300 epochs, the highest number of epochs used in this project. Firstly, the CE standalone loss was the only loss to be trained for more than 100 epochs. This was done due to the fact that the implementation of the loss comes as a part of PyTorch, which meant that it was easy to implement and use, as well as being extremely fast compared to the other losses. The continuity loss utilized the Gudhi library, which is a C++ library for topological data analysis, meaning it only runs on the CPU, instead of the GPU. When purely on the GPU, all PyTorch operations have their CUDA equivalents, allowing for extremely efficient use of the GPU's many cores. This is not the case for the Gudhi library, which means that the continuity loss is significantly slower than the CE loss, and therefore it was only trained for 100 epochs, still taking more than 5 times as long to train as the CE loss for 300 epochs.

With the best model being the one that is trained the longest, it is clear that the training time and number of epochs have a significant impact on the model's performance. However, to show that this is not the only factor, DeepLab was trained for 5000 epochs as a standalone test. This expansive test was still completed in less time than the continuity loss for 100 epochs. The results are shown in @tab:longer_training, where it can be seen that the mIoU for 1000 epochs is indeed higher than for 300 epochs, rising from 0.4497 to 0.4545, going above the apparent 0.45 ceiling, if only slightly. However, the 5000 epochs checkpoint has shown a clear decrease in mIoU performance, dropping to 0.4501. This is also relfected in the per-class IoU, where the values also dropped, outside a miniscule increase in class 1 #ball(color.rgb("#550f6b")). This goes to show that the models used in this project, did not simply need more training time to achieve better performance.

#let tab = [
  #figure(
    {
      tablec(
        columns: 8,
        alignment: (x, y) => (left, center, center, center, center, center, center, center).at(x),
        header: table.header(
          [Model], [Epoch], [Class 0], [Class 1], [Class 2],
          [Class 3], [Class 4], [mIoU $arrow.t$]
        ),

        [DeepLabV3+], [1000], [0.9767], [0.3483], [0.3379], [0.3409], [0.2688], [0.4545],
        [DeepLabV3+], [5000], [0.9758], [0.3485], [0.337],  [0.3313], [0.258],  [0.4501],

        []
      )
    },
    caption: [Per-class IoU and mean IoU for the four models trained with plain CE loss at 1000 and 5000 epochs.]
  ) <tab:longer_training>
]

#tab

Another test was conducted to see if the results would improve if the combination of the CE and cold map losses was swapped around, such that cold map was the main driver for the starting phase of training and slowly changing to CE. The dynamic values for $alpha$ were set to $alpha_"hi" = 0.9$, $alpha_"lo" = 0.5$, and $T_"warm" = 10$. So CE would have some influence in the beginning, which would increase as training progressed. This results of this are shown in @fig:cmap-ce. At the 10th epoch, the results do look very promising, as they look just like the cold map standalone results, but with the paths being correctly labelled. It does, however, still show signs of the artifacts introduced by the cold map loss earlier, in that it continues to pad certain paths will unrelated pixels. This is particularly seen in the bottom images of @fig:cmap-ce#subfigure("a-d") where the blob of incorrectly labelled pixels are persistent across all epochs. This blob is non-existent in the top images, showing that the method works, but also shows potentially worrying artifacts produced in certain scenarios.

#let base = "../../figures/img/discussion/"

#let cmap-ce_e10_test1 = image(base + "deeplab_cmap-ce_e10_test1.png")
#let cmap-ce_e10_test2 = image(base + "deeplab_cmap-ce_e10_test2.png")

#let cmap-ce_e20_test1 = image(base + "deeplab_cmap-ce_e20_test1.png")
#let cmap-ce_e20_test2 = image(base + "deeplab_cmap-ce_e20_test2.png")

#let cmap-ce_e50_test1 = image(base + "deeplab_cmap-ce_e50_test1.png")
#let cmap-ce_e50_test2 = image(base + "deeplab_cmap-ce_e50_test2.png")

#let cmap-ce_e100_test1 = image(base + "deeplab_cmap-ce_e100_test1.png")
#let cmap-ce_e100_test2 = image(base + "deeplab_cmap-ce_e100_test2.png")

#let cmap-ce_acc_graph = image(base + "deeplab_cmap-ce_test_graph.png")
#let cmap-ce_loss_graph = image(base + "deeplab_cmap-ce_train_graph.png")

#std-block(breakable: false)[
  #figure(
    stack(
      grid(columns: (1fr, 1fr, 1fr, 1fr), column-gutter: 0mm,
        cmap-ce_e10_test1, cmap-ce_e20_test1, cmap-ce_e50_test1, cmap-ce_e100_test1,
        cmap-ce_e10_test2, cmap-ce_e20_test2, cmap-ce_e50_test2, cmap-ce_e100_test2,
        [#subfigure("(a)") 10th epoch.], [#subfigure("(b)") 20th epoch.],
        [#subfigure("(c)") 50th epoch.], [#subfigure("(d)") 100th epoch.]
      ), [#v(0.5em)],
      grid(columns: (1fr, 1fr), column-gutter: 0mm,
        cmap-ce_loss_graph, cmap-ce_acc_graph,
        [#subfigure("(e)") Loss graph.], [#subfigure("(f)") Accuracy graph.]
      ),
    ),
    caption: [Qualitative, #subfigure("a-d"), and quantitative, #subfigure("e-f"), results of the DeepLabV3+ model trained with the cold map loss as the main driver in early stages. The rows use the same images as detailed in @c5:results.]
  ) <fig:cmap-ce>
]

At the 100th epoch, when the CE loss has had a large influence for a while, the results are looking a lot like those in @c5:results, which is not a good sign. Many of the same artifacts reappear at this stage, such as split paths and disconnected components. Furthermore, the graphs from this training run, shown in @fig:cmap-ce#subfigure("e-f"), show that the loss is not decreasing as expected, but rather increasing, while the accuracy is reaching a plateau. Once again, this is a clear indication that the models are not generalizing well to the data, and the order in which the losses are applied does not seem to have a significant impact on the results. This is further supported by the fact that the 100th epoch results are very similar to those of the 100 epochs of CE loss alone. This outcome also gives some insight into #RQ(2), in that the cold map loss is not strong enough to keep the CE loss from overfitting the models to specific paths present in the dataset.

Before looking at the transformer models' behaviour, a brief note on the apparent divergence between loss and accuracy is warranted. As shown throughout the graphs in @c5:results, the loss and accuracy values rarely follow the same trend, in that one may increase while the other decreases or plateaus. This is particularly apparent after the cosine annealing restarts. This is a common phenomenon in deep learning, where the loss function may not always reflect the model's performance on the task at hand. In this case, it is likely that the loss function or optimizer is not capturing the nuances of the task, leading to a divergence between the loss and accuracy values. 

Now, looking back at the results of the transformer-based models in @c5:results, it is clear that they do not perform as well as the convolution-based models. This is particularly evident in the mIoU values, where the transformer models have a significantly lower mIoU than the convolution-based models. However, their training did appear to run smoothly with just the CE loss, but saw a significantly more unstable training process when combined with the topology-based losses. This is very likely due to the fact that transformers process the input data through their self-attention mechanisms, which can dilute, or even completely ignore, the spatial relationships between pixels in the input image. Therefore, introducing a loss focused on the topology, i.e. spatial relations in the image, can lead to the model not being able to learn the desired features effectively. This incompatibility may result in the topology losses doing seemingly random changes to the transformers' weights, ultimately leading the models to perform worse than if they had been trained without. This does provide insight into answering #RQ(1), in that it shows that the convolution-based models play more nicely with the topology-based loss functions, and given a task where the topology is important, such as path planning, convolution-based models are likely to perform better than transformer-based models.

// - Longer training results: #checked
//   - 1000 epochs: Overall mIoU = 0.4545, Per-class mIoU = [0.9767, 0.3483, 0.3379, 0.3409, 0.2688]
//   - 5000 epochs: Overall mIoU = 0.4501, Per-class mIoU = [0.9758, 0.3485, 0.337, 0.3313, 0.258]
// - Training dynamics: examine the results of models trained with cmap first (DeepLab cmap→ce). #checked
// - Transformer sensitivity: explore why ViT and Swin architectures perform poorly with continuity-based losses. #checked
// - Loss vs. accuracy divergence: explain why a rising loss function can still coexist with decent accuracy. #checked
// - #RQ(1): The comparison between transformer models (ViT, Swin) and convolutional models (U-Net, DeepLab) in terms of their compatibility with continuity loss functions provides direct insight into architectural trade-offs in accuracy and optimization. #checked
// - #RQ(2): Provides further insight into why certain loss functions (e.g., continuity loss with transformer architectures) may not yield expected gains, supporting the case for loss function refinement. #checked


== Broader Implementation Considerations #checked <c6:implementation>

One of the main considerations made thus far with regard to the implementation of the proposed system, is the trade-off between cloud and onboard inference. This will also give some insight into #RQ(1), where the efficiency of the models is noted. There are two main differences between cloud and on-board inference, both with their up- and downsides. The main difference is cloud inference having access to much more powerful hardware, which allows for larger models to be used. This is particularly relevant for the transformer-based models, which tend to require more memory and computational power than convolutional models, especially when used as backbones to larger models. Cloud computing does come with its own set of challenges, however, such as the need for vehicles making requests to have a constant connection to the cloud, which may not always be available in all environments. 

However, if the inference is handled on-board, then the vehicle still needs a stable enough connection to fetch satellite images, so this aspect of the considerations is not entirely relevant. The main difference is that the cloud can handle larger models, which may lead to better performance, but at the cost of increased latency. The satellite imagery fetching itself introduces a latency of around 250 ms, which can be added to the times seen in @tab:inference_times. This table shows the inference times of the four models when run on the same hardware, where the U-Net model has the lowest inference time at 4.12 ms, followed by ViT at 9.48 ms, DeepLabV3+ at 11.01 ms, and Swin at 36.29 ms. These are all very low times, and are likely to be even lower on the inference-optimized hardware used in AVs. However, as shown in the table as well, the time from ignition to the first usable frame is significantly higher. This time is measured to simulate the time it takes for the system to initialize, when a vehicle is started, to when the first usable frame is produced. These values do not take into account the time it takes for the satellite image to be fetched, nor the time it takes for the on-board system to actually power up. These values make a case for the use of cloud computing, where the system can be running continuously, allowing for a much lower latency when the vehicle is started. Furthermore, cloud computing saves a lot of resources on the vehicle itself, in that it only needs the vehicle to make API requests to the cloud. In conclusion, the choice between cloud and onboard inference depends on the specific use case and the available infrastructure. If low latency is a priority, then onboard inference may be the better option, but if the models are too large or complex to run efficiently on the vehicle's hardware, then cloud inference may be necessary.


#let tab = [
  #figure(
    {
      tablec(
        columns: (auto, 1fr, 1fr, 1fr, 1fr),
        alignment: (x, y) => (left, center, center, center, center).at(x),
        header: table.header(
          [], [DeepLabV3+], [U-Net], [ViT], [Swin]
        ),

        [Inference Time (ms)], [11.012], [4.124], [9.476], [36.292],
        [Ignition to First Usable Frame (ms)], [4163.134], [3987.752], [8540.016], [11106.948],

        []
      )
    },
    caption: [The average inference times, both for a ready system and a newly ignited system, for the four models used in this project. These times are the average of 5 runs for each model.]
  ) <tab:inference_times>
]

#tab

Early in this project, reinforcement learning (RL) was considered as an alternative to the proposed method. The idea was to use an RL model which would be given a specific set of rules to follow, which it would then learn to follow by trial and error. This would allow the model to learn how to navigate intersections in a more dynamic way, as it would be able to adapt to different situations and learn from its mistakes. The paths it travels, or the pixels it traverses, would then be extracted and used as the generated path. Going from the employed supervised learning approach to an RL approach would require a significant paradigm shift, as the method for training the model is vastly different. Due to limitations in time and manpower, this approach was not pursued further, but it is still an interesting avenue for future work. 

Some considerations towards employing some post-processing techniques were also made. First, is the idea of developing an algorithm that will bridge the gaps between any disconnected components in the generated paths. By doing this, the idea of training the models with loss functions that are not focused on the topology of the paths, may be more realistic, as the post-processing will ensure that the paths are connected. Second, is the idea of skeletonization. This is a technique that reduces the number of points in a line, making it more narrow and thus much more precise for a vehicle to follow. This may prove particularly useful in the case of the transformer-based models, which tend to produce wider paths than the convolutional models. 

Finally, much of this project was made possible thanks to the PyTorch framework, which offered a lot of easy-to-use functionality for the implementation and training of the models. The framework greatly simplified the process of implementing not only the models, but also the topology-based loss functions, as all standard operations are automatically differentiable and can be used in the training process. Python is, however, notoriously slow despite its ease of use and popularity. Therefore, other languages were briefly considered for this project as well, and is definitely an avenue for future work. C/C++ were in the first consideration, as they are widely used languages and are extremely lightweight due to them being very low-level. Fortran was also considered, largely due to the fact that it seemed to gain a lot of attention at the NVIDIA GTC 2025 conference, highlighting how industry relevant it still is. As with C/C++, Fortran is also a low-level language, which means that it can be used to write highly efficient code. The focus on these languages at the conference also highlights the shift happening within the AI industry, where the focus is moving towards more efficient and lightweight implementations of AI models. 

// - Skeletonization: employ post-processing to ensure paths are 1-pixel wide.
// - Language considerations: C/C++ and Fortran were considered for high-performance alternatives.
// - Model alternatives: Reinforcement Learning considered for dynamic decision-making tasks. #checked
// - Inference environment: #checked
//   - Average inference times (ms): [11.012, 4.124, 9.476, 36.292] #checked
//   - Satellite request latency: \~250 ms #checked
//   - Time from ignition to first usable frame (ms): [4163.134, 3987.752, 8540.016, 11106.948] #checked
//   - Cloud deployment benefits: allows for persistent availability, reduced local hardware requirements, and lower latency on start-up. #checked
// - #RQ(1): Covers inference times and implementation aspects (cloud vs. onboard), which ties into optimizing for efficiency. #checked

== Domain Transfer and Industrial Relevance #checked <c6:domain-transfer>

The method developed in this project has primarily been designed for the use in AVs, specifically for the task of predicting paths through intersections based on satellite imagery. However, this specified task can be seen as a specific instance of a broader problem: predicting spatially viable paths from static images. This core concept has potential relevance in several other domains, which can be explored as speculative applications of the method. 

In the context of warehouse robotics, the system could be adapted to predict efficient routes through dynamically configured storage layouts, such as floor plans or occupancy grids. This would allow for more efficient navigation of warehouse robots, which is particularly relevant in production logistics. The method could be retrained when the warehouse layout changes, assuming that overhead maps are available, which is realistic in many automated facilities. 

In autonomous racing, the method could be used to predict aggressive yet feasible racing lines from track images or schematic representations. In the context of racing, intersections are not as relevant, but the method could still be used to predict various paths through corners and other track features, such as a more aggressive line or a safer, slower line. 

The general AV case extends the method from intersections to broader environments, such as urban driving or off-road vehicles. Urban driving still benefits from map-based segmentation, as showcased in this thesis, while off-road vehicles may use satellite-style imagery or drone data. More advanced approaches to this method could even help automata navigate off-road environments, down to where they should place their wheels or feet, if the resolution is high enough. Extending this even further, the method could be used for end-to-end planning in unfamiliar environments with a simple drone taking pictures of the environment to cross, then pre-planning every step of the way. This further highlights the scalability and modularity of the method, as it can be integrated as a module for pre-planning in unfamiliar environments.

Furthermore, it can be deployed in other scenarios where AVs are the prime actor. For example, underwater robots or deep-sea autonomous underwater vehicles (AUVs) could use sonar maps or pre-mapped seabeds to plan safe navigation paths. This is particularly interesting due to the absence of live perception in deep-sea environments, making inference on prior data a practical solution, akin to the cold maps used in this project. 

In summary, this highlights how the method can be used in various domains and industries, from warehouse robotics to autonomous racing, general AV systems, and deep-sea automation. The core strength of the approach lies in its ability to operate on static imagery without requiring real-time sensor data, making it particularly valuable in environments where live perception is limited or unreliable. The alternative, where a dependency on cameras for positioning, is also possible, as discussed in the next section. The modularity of the system also allows for domain-specific adaptations while maintaining the fundamental path prediction framework. However, successful domain transfer would likely require retraining on domain-specific datasets and potentially adjusting the model architecture to account for different spatial scales, image resolutions, and environmental constraints specific to each application domain.

// Though the method was designed for autonomous vehicle intersection traversal, its core concept—predicting spatially viable paths from a static image—has potential relevance in several other domains.

// In warehouse robotics, the system could be adapted to floor plans or occupancy grids, predicting efficient routes through dynamically configured storage layouts. Could be retrained when warehouse layout changes. Assumes overhead map availability, which is realistic in many automated facilities. Relevant due to strong ties between robotics and production logistics.

// In autonomous racing, predicting aggressive yet feasible racing lines from track images or schematic representations aligns with the model's strengths. Could augment traditional planning pipelines for edge-case awareness or act as a fast planner in simulation. Requires faster inference, but this is already explored in the timing section.

// The general AV case extends the method from intersections to broader environments. Urban driving still benefits from map-based segmentation, and off-road vehicles may use satellite-style imagery or drone data. Could be integrated as a module for pre-planning in unfamiliar environments or low-connectivity zones. Scalability, model modularity, and lack of runtime training are industrially appealing.

// For underwater robots or deep-sea AUVs, sonar maps or pre-mapped seabeds can substitute for satellite imagery. Here, pre-trained models can help plan safe navigation paths, similar to how seabed features are navigated during pipeline inspections. Particularly interesting due to the absence of live perception in deep-sea environments, making inference-on-prior-data a practical solution.


== Robustness and Domain Generalization #checked <c6:robustness>

One thing to consider before expanding the method to other domains is the goal of achieving great domain generalization. In this project, this goal has been partially achieved, as the method is able to create fairly precise predictions of paths through intersections, depending on the model used. A greater level of generalization is still needed, however, as the models are still heavily overfitting to the training data, as shown in the results chapter. This also gives some insight into #RQ(3), which asks whether it is possible to create a dataset that allows for the training of a model, such that the data is not too stringent to a singular path. While this is partly achieved in that the models can generate paths in any image given, the current dataset does not allow for the models to gain a general domain understanding.

This lacking capability of the dataset is particularly apparent when looking at the results of the transformer-based models, which generally struggle to generate outputs that are as precise as the convolutional models. This is, as mentioned briefly earlier, likely due to the fact that transformers dilute the fine-grained spatial relationships in an image, meaning they have difficulty learning small structures and boundaries. The different inductive biases of these architectures—where convolutional models inherently assume spatial locality and translation invariance, while transformers rely on global attention mechanisms—significantly affect their ability to handle topology-aware tasks. This is reverently apparent in the results, where the transformer-based models tend to produce wider paths than the road allows. To improve the robustness of the use of transformers, it is likely that some kind of hybrid system would be needed, where the transformer is used to generate a rough path, which is then refined by a convolutional model. Or by simply combining transformers with convolutional layers.

Once models are created that are good enough to create clearly defined paths through intersections, the need for a way to ensure that a path is being followed is paramount. There exists work that utilize the #acr("BEV") cameras to localize the vehicle in a given image @bev. With this combination of methods, however, arises the need for robustness in the form of handling seasonal changes. For example, if the image used to predict the path is from a clear, sunny day and the system creates a path through it nicely, then the localization may not be able to accurately determine the vehicle's position if the intersection is in reality covered in snow, obscuring the road markings. Therefore, to increase both robustness and domain generalization, the models used should be trained on a variety of images, maybe even creating seasonal variants of the dataset. This does pose its own set of challenges, as satellite imagery is not always available for every season. Work has been done to add synthetic attributes to images. The attribution manipulation framework proposed by Karacan #etal @season, allows for the addition of synthetic attributes to images, such as snow or rain, which could be used to augment the dataset and create seasonal variants. This would allow for the models to be trained on a wider variety of images, increasing their robustness and generalization capabilities.





// - Domain extension: speculative applicability to warehouse robots, autonomous racing, general AV systems, and deep-sea automation. #checked
// - Seasonal robustness: markings may disappear in snow; suggest periodic retraining or season-specific model variants. #checked
// - #RQ(2): Touches on the structural priors of models, which relate to how well loss functions can exploit or align with model inductive biases.
//   - Structural priors in transformers: investigate how ViT and Swin handle (or fail to handle) spatial and structural assumptions. (As noted, they struggle with topology losses as they dilute spatial relationships.)
// - #RQ(3): Includes thoughts on domain generalization, speculative applications, and the challenges of applying a model trained on limited data to broader settings—all of which hinge on the flexibility of the dataset. #checked


== Societal and Ethical Considerations #checked <c6:societal>

Finally, if the proposed system is to be deployed in real-world scenarios, it is important to consider the societal and ethical implications of such a system. In general, it is believed that the system can have an overall positive impact on many aspects of society, both for pedestrians, drivers, and manufacturers. For manufacturers, the system can help improve the performance of their vehicles, as the system can help them navigate intersections more efficiently. This will allow them to keep pushing for higher levels of autonomy, as they can use the system to keep improving their vehicles' capabilities.

For the owners of these vehicles, the system will add transparency to the vehicle's decision-making process, as it will be able to explain why it is taking a certain path through an intersection. This will strengthen the trust between the vehicle's manufacturer and the owner. For pedestrians, it means that the vehicles will be able to navigate intersections more safely, and, for a pedestrian's comfort, more predictably, as the vehicles will follow a more logical and smooth path through intersections.

Legally, the system will need to comply with various regulations related to critical infrastructure, such as road markings and traffic signs. The system will also need to ensure that it does not infringe on any copyrights or licenses related to the satellite imagery used for training and inference. This is particularly relevant in the context of using services like Google Maps, which have specific terms of use regarding their imagery. Furthermore, considerations should be made in case the system contributes to an accident, such as the generated path leading the vehicle down the wrong side of the road or thinking it can make a turn despite not being in a turning lane. Ethically, the system should also consider the privacy of individuals, as the satellite imagery used for training and inference may contain sensitive information, such as private properties or people. 

// - Positive impacts: potential improvements in traffic efficiency, safety, and environmental sustainability.
// - Legal and ethical challenges:
//   - Imagery licensing and permitted use
//   - Privacy-preserving techniques (e.g., blurring of sensitive areas)
//   - Compliance with regulations related to critical infrastructure
// - #RQ(3): May briefly connect to this question through ethical implications of deploying systems trained on potentially biased or narrow datasets.
  
// Use skeletonization to reduce the number of points in the line and make it 1 pixel wide
// other considered languages: C/C++, Fortran
// other models: RL

// thought experiment on how to implement for other domains (warehouse robots, racing, overall AV, deep sea automation). speculative

// examiner from the production engineering world

// Robustness to seasonal changes (snow-covered markings disappear) – argue for periodic retraining or synthetic augmentation. Or maybe even use different models used for different seasons/conditions.

// on-board vs cloud processing
// Average inference times: [11.012  4.124  9.476 36.292]
// Sat time: ~250ms
// Time from ignition to first frame: [ 4163.134  3987.752  8540.016 11106.948]
// Justify cloud as it can be ready at all times and API call is faster. Less resource intensive.

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


/// Societal implications
// Impact on traffic flow, safety, and the environment
// legal/ethical: imagery licensing, privacy masking, critical-infrastructure restrictions.

// == Ablation <c6:ablation>
// MAYBE, if time allows

// Hyperparameters, scheduler (cosann vs exp), optimizer (adam vs adamw), weight initialization