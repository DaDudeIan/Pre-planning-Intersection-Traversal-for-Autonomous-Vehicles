#import "../../lib/mod.typ": *
= Abstract #checked <c0:abstract>

Autonomous vehicles (AVs) still struggle to negotiate intersections reliably because on-board perception can falter and infrastructure support (e.g., V2X) is not yet ubiquitous.
This thesis investigates a complementary strategy: pre-planning the exact lane-level path through an intersection before arrival using deep-learning models that read satellite imagery.

A bespoke dataset of 112 real European intersections was compiled, each annotated with per-pixel turn masks and a novel cold map overlay that softly penalises distance from the reference path, and was inflated through extensive colour, geometric and zoom augmentations to combat data scarcity.
Four architectures—DeepLabV3+, U-Net, Vision Transformer (ViT) and Swin Transformer—were trained under a unified pipeline that paired standard Cross-Entropy with two topology-aware losses: the new cold map loss and a persistent-homology-based continuity loss.
Training used an AdamW optimiser with cosine-annealing warm restarts to encourage exploration of the loss landscape.

On an unseen validation split, DeepLabV3+ with plain Cross-Entropy reached the highest mean Intersection-over-Union (mIoU = 0.45) across the five designated classes, while U-Net offered the fastest inference ($approx 4"ms"$ on the test platform).
Topology-aware losses produced visibly cleaner, more connected trajectories for the convolutional models, but did not raise mIoU; transformer models under-performed, reflecting their larger data appetite. Memory footprints stayed below 350 MB, aligning with typical EV compute budgets.

The study demonstrates that coarse satellite views contain enough latent structure for deep networks to draft feasible, lane-respecting intersection paths in milliseconds, offering AV stacks a sensor-agnostic “look-ahead” capability. Key bottlenecks are dataset scale and diversity; future work should expand training imagery, refine topology losses and integrate automatic orientation handling to move from proof-of-concept toward deployment.
