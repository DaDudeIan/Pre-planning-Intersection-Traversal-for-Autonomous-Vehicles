#import "../../lib/mod.typ": *

= Related Work <c3:related-work>
// works that already utilize satellite imagery i.e. works that use sat images for autonomous navigation

== Path-planning <c3s3.1:path-planning>
// Traditional: A*, RRT*, Dijkstra, D*
// DL for trajectory prediction, imitation learning
// Map-based vs sensor-based

Path-planning is the task of having a certain amount of knowledge about the environment and finding a path from a starting point to a goal point. This task is one of the most fundamental tasks in the field of robotics and autonomous navigation, and has thus has a long history of improvement and evolution. 

One of the first algorithms to be used for path-planning is the Dijkstra algorithm from 1959 @dijkstra1959-cg, which is a graph search algorithm that finds the shortest path between nodes in a graph. The A\* algorithm is another popular algorithm from 1968 that is used for path-planning, and is a combination of Dijkstra's algorithm and a heuristic function that estimates the cost of the cheapest path from a node to the goal node @a-star. Some years later, the D\* algorithm was introduced in 1994, which is an incremental search algorithm that finds the shortest path between nodes in a graph, and is an improvement over the A\* algorithm @d-star. D\* has since become a very popular algorithm for path-planning in robotics, with improved alternatives like Focused D\* the year after @focused-d-star and D\* Lite from 2005 @d-star-lite proving use in real-world applications.

The concept of #acrpl("APF") was introduced in 1986 @apf. It assumes some repulsive field around obstacles to avoid and a pulling force towards the goal, resulting in autonomous robots navigating towards the goal while avoiding obstacles. This method is particularly effective in dynamic environments, where the obstacles are moving. It is, however, very prone to local minima and situations where it might get trapped @apf-boo, significantly reducing its effectiveness in complex environments, such as a tight hallway where the robot might fit but the calculated repulsive force is too great or if it encounters a dead-end or U-shaped obstacle, leading it to loop infinitely. Combined with other global path-planning algorithms, it has shown considerable success, especially in the field of swarm robotics @apf-swarm1@apf-swarm2. 

The #acr("RRT") algorithm was introduced in 1998 @rrt, and is a popular algorithm for path-planning in robotics. It is a randomized algorithm that builds a tree of possible paths from the starting point to the goal point, and is particularly useful in high-dimensional spaces. A node is randomly chosen from the initial point. The intermediate node is then determined based on the movement direction and maximum section length. If obstacles are detected, the route in that direction is ignored. Otherwise, a new random point is selected. The RRT\* algorithm was introduced in 2011 @rrt-star, improving on the original with two small but significant modifications: a cost function that takes into account the distance between nodes and a re-wiring step that allows the tree to be restructured to find a better path. It has shown great usage in real-world applications regarding #acrpl("AUV") @rrt-auv1@rrt-auv2, despite challenges regarding the need for information about large areas @rrt-auv3.

Other areas of research in path-planning include #acrpl("GA") and #acr("FL"). #acr("GA") @ga is inspired by the process of natural selection where only the fittest organisms survive. Generally the algorithm works by generating a random population of solutions, and then selecting the most efficient ones by using some cost function. Then these selected solutions go through the crossover process where they are combined and mutated to generate new solutions. #acr("FL") is another old method from 1965 used for path-planning @fuzzy_sets@fuzzy_algs. It depends on functions used in fuzzification, inference, and defuzzification. These functions are based on a descriptive classification of the input data, such as low, medium, or high collision risk. Based on the defuzzification process, the robot decides on the best path to take.

#acrpl("NN") are also finding their usage in the field. #acrpl("NN") are made to imitate the human brain's innate ability to learn. They are trained on data and learn how to react to it. They are used in the field, not necessarily for path-planning explicitly, but more in conjunction with other algorithms that use their output as input. I.e. a #acr("NN") might be able to tell the controller where some obstacle is, meaning it is giving a helping hand to algorithms like #acr("APF"). Akin to #acr("APF"), #acr("RL") models are taught to react to their surroundings, driving towards a goal and being rewarded and penalized for the actions that it takes, like how #acr("APF") is moving towards a goal and avoiding obstacles due to the repulsive forces.

In summary, the evolution of path-planning $dash$ from early graph search methods like Dijkstra and A\* to more adaptive techniques such as D\*, RRT, and learning-based models $dash$ illustrates a steady push toward efficiency and robustness. Approaches like #acrpl("APF"), #acrpl("GA"), and #acr("FL") add further flexibility, each with its own trade-offs. Together, these methods highlight the ongoing effort to balance computational efficiency with real-world challenges.


== Intersection Management <c3s3.2:intersection-management>
// AIM, V2X
// Multi-agent RL, Cooperative learning
// Simulation (sim-to-real)
// 

// == On-board vs Cloud Computing <c3s3.3:edge-cloud> // Maybe discussion (theoretical) 