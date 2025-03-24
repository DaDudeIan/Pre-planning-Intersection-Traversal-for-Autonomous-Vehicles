#import "../../lib/mod.typ": *

// = Background <c2:background>

== Autonomous Vehicles <c2s2.1:av>
// Levels of autonomy (SAE 0-5) and achievements (Tesla, Waymo, Cruise)
// https://www.sae.org/standards/content/j3016_202104/
// Challenges experienced by AVs (adverse weather and its effects on sensors and perception related to intersections.) Lane-level localization

// Current methods for understanding real-world
// - sensors (lidar, radar, cameras)
// - sensor fusion (kalman filters, bayesian fusion), V2X (Car2X, DSRC)
// - control systems (PID, MPC, LQR, reinforcement learning, gbp planner) (conjunction with path planning, robot control problem)
// - decision making (rule-based, behavioural cloning, imitation learning, reinforcement learning)

== Deep Learning <c2s2.2:dl>
// Fundamentals
// - MLP, backpropagation, gradient descent
// - Activation functions, optimizers, loss functions, regularization, scheduler, logits
// Types: Reinforcement Learning, Supervised Learning, Unsupervised Learning, transfer learning, self-supervised learning
// point of first contact with AI as lead-in to CNNs
// Recurrent Neural Networks and LSTMs (sequential decision-making), 
// Graph NN (road topology, scene graph modelling)

=== Computer Vision <c2s2.2.1:cv>
// Classic CV tasks: object detection, segmentation
// Conv and CNNs, Vision Transformers ViT (Attention), Swin, U-Net, (conditional) GANs
// Bird's Eye View (BEV) models, Liquid NN, graph NNs
// Lane Detection
// specific papers: DiffusionDrive
// 
// Fast inference is not a critical goal, but accuracy is.

=== Datasets <c2s2.2.2:datasets>
// Existing datasets for AVs: waymo, nuScenes, Argoverse
// lane-level datasets: ONCE, OpenLane, 
// Road extraction: DeepGlobe, SpaceNet
// Trajectory prediction: INTERACTION, ApolloScape
// 
// Training: introduce occlusions

== Satellite Imagery <c2s2.3:si>
// Google Maps Static API, Azure Maps, Sentinel, OpenStreetMap
// Works even when cloudy (non-reliance on live sat images)
// Resolution
// HD maps

== Path Drawing <c2s2.4:path_drawing>
// Waypoints, trajectory smoothing, Bézier curves, splines

== Pose Estimation <c2s2.5:pose_estimation>
// Potential for future work
// ensure correctness of vehicle following path