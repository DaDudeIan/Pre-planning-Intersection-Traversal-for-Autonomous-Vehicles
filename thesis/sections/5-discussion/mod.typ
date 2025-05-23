#import "../../lib/mod.typ": *
= Discussion <c6:Discussion>

This section presents the discussion of the results, methods, and broader implications of the work carried out in this thesis. It begins with the integration of the proposed system into real-world infrastructure, including its compatibility with V2X communication, considerations related to ISO 26262 compliance, and how it should actually work with current systems active in vehicles. Following this, the notable shortcomings of the project are outlined, including performance limitations, data dependencies, and the limited improvements observed with topological loss functions. After this, several technical insights are discussed, such as the behaviour of transformer-based models, the relationship between loss and accuracy, and the outcomes of extended training runs. The next section covers implementation-related considerations, such as the trade-offs between cloud and onboard inference, inference times, and post-processing strategies. Generalization and robustness are then explored, with a focus on seasonal changes, cross-domain applicability, and the inductive biases of different model architectures. Finally, the chapter concludes with a discussion of the societal and ethical aspects associated with deploying such a system, including privacy, legality, and the potential impact on traffic and the environment.


== Integration with existing systems <c6:integration>

- Integration with V2X: rely on V2X communication when available; otherwise, fall back to the onboard model.
- Memory footprint: on-board memory is not a concern, as models are designed for inference and are lightweight.
- GPS limitations: GPS alone does not provide sufficient information for precise path planning.
- Rotation estimation from GPS: determine vehicle orientation using the vector between the current position and the intersection center.
- Compliance with ISO 26262: implement a hand-over strategy that aligns with functional safety standards.

// rely on v2x when available, otherwise this method
// On-board memory footprint of each model (hardly a problem as models do not require training. Computers optimized for inference.)

// GPS not enough
// How to find the angle to rotate? When driving with a GPS, use coordinates to the centre. Find angle by finding the angle between the two points. 
// 
// Hand-over strategy in compliance with IDO 26262

== Project Limitations and Challenges <c6:shortcomings>
- Dependency on fresh satellite imagery: the system requires up-to-date imagery to remain reliable, especially in dynamic environments such as construction zones.
- Performance plateau: the overall mIoU stagnated below 0.45, indicating a need for more diverse and extensive training data.
- Visual misclassification: include examples of predictions that appear correct but fail under safety-critical scrutiny.
- Limited gains from topology-aware losses: the expected benefits of continuity and branching losses were not consistently observed.
- Class imbalance: justify the use of Cross-Entropy and Binary Cross-Entropy; reflect on the potential benefits of Focal Loss and Dice Loss.
- #RQ(2): Evaluates the performance of the topology-based loss functions (e.g., continuity loss), highlighting their limited effectiveness and raising important considerations about how to better capture path similarity without pixel-level matching.
- #RQ(3): Discusses class imbalance and the need for more varied data, which relates to avoiding overfitting to a single type of path and supporting generalization.


// Notable shortcomings of project: requires latest satellite images to be completely useful. (like construction zones, etc)

// mIoU never getting above 0.45, pointing out need for more data.

// illustrate the “looks good but fails safety check” issue noted in the results discussion. (Include zoom in on different images.)

// Lacking improvements introduced by topological loss function. 

// Class imbalance. Justify use of CE and BCE. Discuss focal and Dice loss.


== Technical Observations and Training Insights <c6:technical-insights>

- Longer training results:
  - 1000 epochs: Overall mIoU = 0.4545, Per-class mIoU = [0.9767, 0.3483, 0.3379, 0.3409, 0.2688]
  - 5000 epochs: Overall mIoU = 0.4501, Per-class mIoU = [0.9758, 0.3485, 0.337, 0.3313, 0.258]
- Training dynamics: examine the results of models trained with cmap first (DeepLab cmap→ce).
- Transformer sensitivity: explore why ViT and Swin architectures perform poorly with continuity-based losses.
- Loss vs. accuracy divergence: explain why a rising loss function can still coexist with decent accuracy.
- #RQ(1): The comparison between transformer models (ViT, Swin) and convolutional models (U-Net, DeepLab) in terms of their compatibility with continuity loss functions provides direct insight into architectural trade-offs in accuracy and optimization.
- #RQ(2): Provides further insight into why certain loss functions (e.g., continuity loss with transformer architectures) may not yield expected gains, supporting the case for loss function refinement.


== Broader Implementation Considerations <c6:implementation>

- Skeletonization: employ post-processing to ensure paths are 1-pixel wide.
- Language considerations: C/C++ and Fortran were considered for high-performance alternatives.
- Model alternatives: Reinforcement Learning considered for dynamic decision-making tasks.
- Inference environment:
  - Average inference times (ms): [11.012, 4.124, 9.476, 36.292]
  - Satellite request latency: \~250 ms
  - Time from ignition to first usable frame (ms): [4163.134, 3987.752, 8540.016, 11106.948]
  - Cloud deployment benefits: allows for persistent availability, reduced local hardware requirements, and lower latency on start-up.
- #RQ(1): Covers inference times and implementation aspects (cloud vs. onboard), which ties into optimizing for efficiency.

== Domain Transfer and Industrial Relevance <c6:domain-transfer>
// Though the method was designed for autonomous vehicle intersection traversal, its core concept—predicting spatially viable paths from a static image—has potential relevance in several other domains.

// In warehouse robotics, the system could be adapted to floor plans or occupancy grids, predicting efficient routes through dynamically configured storage layouts. Could be retrained when warehouse layout changes. Assumes overhead map availability, which is realistic in many automated facilities. Relevant due to strong ties between robotics and production logistics.

// In autonomous racing, predicting aggressive yet feasible racing lines from track images or schematic representations aligns with the model's strengths. Could augment traditional planning pipelines for edge-case awareness or act as a fast planner in simulation. Requires faster inference, but this is already explored in the timing section.

// The general AV case extends the method from intersections to broader environments. Urban driving still benefits from map-based segmentation, and off-road vehicles may use satellite-style imagery or drone data. Could be integrated as a module for pre-planning in unfamiliar environments or low-connectivity zones. Scalability, model modularity, and lack of runtime training are industrially appealing.

// For underwater robots or deep-sea AUVs, sonar maps or pre-mapped seabeds can substitute for satellite imagery. Here, pre-trained models can help plan safe navigation paths, similar to how seabed features are navigated during pipeline inspections. Particularly interesting due to the absence of live perception in deep-sea environments, making inference-on-prior-data a practical solution.

// Common theme: model’s modular design and inference-only requirement makes it attractive for embedded systems in production environments. Fits well into system architectures where retraining is costly or impractical. This modularity and reusability is often sought after in production engineering.


== Robustness and Domain Generalization <c6:robustness>
- Seasonal robustness: markings may disappear in snow; suggest periodic retraining or season-specific model variants.
- Domain extension: speculative applicability to warehouse robots, autonomous racing, general AV systems, and deep-sea automation.
- Structural priors in transformers: investigate how ViT and Swin handle (or fail to handle) spatial and structural assumptions.
- #RQ(2): Touches on the structural priors of models, which relate to how well loss functions can exploit or align with model inductive biases.
- #RQ(3): Includes thoughts on domain generalization, speculative applications, and the challenges of applying a model trained on limited data to broader settings—all of which hinge on the flexibility of the dataset.


== Societal and Ethical Considerations <c6:societal>
- Positive impacts: potential improvements in traffic efficiency, safety, and environmental sustainability.
- Legal and ethical challenges:
  - Imagery licensing and permitted use
  - Privacy-preserving techniques (e.g., blurring of sensitive areas)
  - Compliance with regulations related to critical infrastructure
- #RQ(3): May briefly connect to this question through ethical implications of deploying systems trained on potentially biased or narrow datasets.
  
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