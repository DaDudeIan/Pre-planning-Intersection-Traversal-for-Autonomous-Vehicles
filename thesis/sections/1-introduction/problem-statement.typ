#import "../../lib/mod.typ": *

// https://www.scribbr.com/research-process/problem-statement/
== Problem Statement <c1s1_1:problem-statement>

// Importance of the problem / Contextualize
Advancements in #acr("AV") technologies have been at the forefront of tech innovations in the #nths(21) century. A key challenge in the development of fully autonomous vehicles, is their ability to handle intersections. Intersections pose a wide variety of challenges to #acrpl("AV"): from those posed by complex structures, to those posed by the unpredictability of human drivers, to faded lines that make it difficult for on-board computer vision system to clearly identify lanes or paths. All of these hinder #acrpl("AV") from reaching their full potential and being able to navigate intersections safely and efficiently.

// Specify gaps in current solutions: Infrastructure dependence (Car2X, AIM), sensor limitations (cam, radar, lidar), 
Current existing solutions are very infrastructure-dependent. The Car2X system by Volkswagen, for example, relies on a network of sensors and communication devices installed in the infrastructure to spread information to vehicles on the road @car2x. #acr("AIM") also relies on infrastructure to provide vehicles with information regarding intersections, with an orchestrator monitoring and managing individual intersections @aim@aim_heuristic@aim_survey, with active development moving towards a more decentralized and distributed approach @aim_dist. Furthermore, reliance on camera-based vision is susceptible to environmental limitations, such as adverse weather, that reduce system reliability.

// Consequences of the problem / Why it matters
The challenges posed by intersections cause major problems for #acr("AV") developers who want to push fully autonomous driving. #acrpl("AV")' inability to properly react to and handle intersections, leads to significant delays in real-world deployment as a consequence of the unreliability experienced by regulators and the general public. If #acrpl("AV") want to enter the market with full self-driving capabilities, full autonomy is a key challenge to be tackled, as it is an essential task experienced when driving. 


// Aim and objectives for project / solution direction
This projects aims to develop a solution that will help #acrpl("AV") to better handle intersections. With the use of #acr("DL") and #acr("CV") technologies, trained on and utilizing satellite imagery, this project aims to train a model that can accurately identify the proper path for an #acr("AV") to travel through an intersection. The system is not meant to replace current systems deployed in #acrpl("AV"), but rather assist the existing systems make better decisions when in self-driving mode and approaching an arbitrary intersection.  