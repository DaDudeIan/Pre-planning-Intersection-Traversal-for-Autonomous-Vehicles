#import "../../lib/mod.typ": *


// = Background <c2:background>

== Autonomous Vehicles <c2s2.1:av>
// Levels of autonomy (SAE 0-5) and achievements (Tesla, Waymo, Cruise)
// https://www.sae.org/standards/content/j3016_202104/
// Challenges experienced by AVs (adverse weather and its effects on sensors and perception related to intersections.) Lane-level localization

Initially released in 2014 by the Society of Automotive Engineers (SAE), the J3016 standard @sae1 defines six levels of driving automation, ranging from Level 0 to Level 5, with the latest revision released in 2021 @sae2. These levels are visualized in @fig:5levels. These levels are further split into two separate categories based on the environment observer; the first three levels are concerned with the human driver being the environment observer, and the latter three levels are concerned with the vehicle being the environment observer named automatic driving system (ADS) features. 

To understand how SAE defines the levels, it is important to gain a high-level overview of how these levels are defined. The document starts by defining the scope of the standard, clearly stating that it "describes [motor] vehicle driving automation systems that perform part or all of the dynamic driving task (DDT) on a sustained basis" #cite(<sae2>, supplement: "p. 4"). This definition excludes any momentary actor systems in place in a car, such as electronic stability control, automatic emergency braking, or lane keeping assistance (LKA). These systems are not considered to be part of the DDT, as they do not perform the driving task on a sustained basis. Finally, three primary actors are defined: the human user, the driving automation system, and other vehicle systems and components.

#let fig = image("../../figures/img/bg/5levels.png")
#let my_brace(n) = [#v(-12pt)$underbrace(#box(width: 80%))$ #v(0mm) Level #str(n)]

#std-block(breakable: false)[
    #figure(
        grid(
        columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
        column-gutter: 10mm,
        inset: 0mm,
        align: (center, center, center, center, center, center),
        grid.cell(fig, colspan: 6),
        [#my_brace(0)], [#my_brace(1)], [#my_brace(2)], [#my_brace(3)], [#my_brace(4)], [#my_brace(5)],
    ),
    caption: [SAE 6 levels of autonomy. Image source: Synopsys @synopsys]
    ) <fig:5levels>
    
]

Of course, many different manufacturers are actively developing their own systems, slowly climbing through the levels of automation, with some being further along than others. A comprehensive overview of the levels will be presented the following along with clear examples of each level. The levels are defined as follows:
- *Level 0*: No Automation. _Manual control. The human performs all driving tasks (steering, acceleration, braking, etc)_#footnote([These italicised definitions are from the infographic shown on the Synopsys blog covering the topic @synopsys.]).
Level 0 autonomy may encompass more cars than one would imagine. This is largely due to the fact, that, as stated, momentary systems are not included as giving a vehicle any form of autonomy. Shown with the visualization in @fig:5levels, this level of autonomy required the user to be in full control at all times during the DDT. That means keeping both hands on the wheel at all times and staying aware of the environment for the duration of the trip. Common systems like emergency braking and lane keeping assistance do not push vehicles with these features any higher than this level. This is due to their unsustained nature. 

Today, while exact numbers are unknown, it is believed that the majority of vehicles on the road today are at this level. Statistics by statista @statista1@statista2, hints that very few vehicles sold before 2014 had any form of autonomy. This is shown by their 2015 statistics, showcasing that 51.3% of vehicles sold that year were Level 0. By 2018 this number had dropped to 24% and by 2023 it was down to just 7%, with a massive shift towards Level 1 coming in at 71%.

- *Level 1*: Driver Assistance. _The vehicle features a single automated system (e.g. it monitors speed through cruise control)_.
The first commercially available adaptive cruise control (ACC) vehicles came from Chrysler in 1958 following the invention of the SpeedoStat @first_cc. They named this system the "Auto-Pilot" and vehicles equipped with it, was some of the first vehicles in history to achieve what would later be called Level 1 autonomy. Soon after, Cadillac adopted the technology for their own vehicles and dubbed it "Cruise Control", which since became the default term for the ACC technology, which is still used today, though usually with the term "adaptive" in front of it.

As shown in the visualization in @fig:5levels, the steering wheel has a dashed out look, and if the pedal were shown, they would be dashed out as well. This is to show that the human driver is not in control of every aspect of the DDT. Per the definition, Level 1 autonomy is defined as an autonomous system working on either the lateral or the longitudinal vehicle motion, which means that it can either control the steering or the acceleration of the vehicle. The human driver is still in control of the other aspect of the DDT. ACC is one of the most common systems elevating vehicles into this level. ACC can in this level not work with other systems, such as automatic steering. For vehicles that do both, end up in Level 2.

- *Level 2*: Partial Automation. _Advanced driver-assistance system (ADAS). The vehicle can perform steering and acceleration. The human still monitors all tasks and can take control at any time_.
Level 2 autonomy is the first level where the vehicle can perform both steering and acceleration. This is done through a combination of systems, such as adaptive cruise control (ACC) and lane-keeping assistance (LKA). The human driver is still in control of the vehicle and must monitor the environment at all times. This level is the most common level with #acrpl("EV"), as all have ACC and LKA systems to a smaller or greater extent and capability. As visualized, the driver must always have at least one hand on the wheel at all times and be aware of the surrounding environment. This is the level where most of the current systems are at, such as Tesla Autopilot, Ford BlueCruise, and GM Super Cruise. Despite only being Level 2, Ford's BlueCruise has been allowed to establish "blue zones" in the EU, where drivers are allowed to remove the hands from the steering wheel when engaged @ford.

At this level, the European version of the Tesla Autopilot belongs, where it's American relative is climbing for level 4. This is despite the fact that Tesla Full Self-Driving (FSD) is legally classified as Level 2 in both regions, but in the US they operate at a higher level due to more permissive testing environments and less restrictive oversight @fsd_level. Thus, laws and regulations create a functional separation at this level, as this is the level currently allowed in EU countries, with the single exception, as of 2022, being the Mercedes S-Class, allowed to operate in Level 3 under strict conditions @autoseu. This allowance has since been given to other manufacturers.

- *Level 3*: Conditional Automation. _Environmental detection capabilities. The vehicle can perform most driving tasks, but human override is still required_.
Level 3 autonomy is characterised by their ability to detect and act upon the environment by themselves. This means that the vehicle can perform most DDTs, but the human driver must still be able to take control at any time. This level is the first level where the vehicle can operate without human intervention in certain situations, such as highway driving in cases where overtaking a slow moving vehicle is possible. This is also the first level to rely on automated systems to monitor the environment. 

An important note as the level keeps rising, is what the SAE calls DDT Fallback. This refers to the action of taking over the DDT as a user, in the event of a DDT performance-relevant system failure or upon operational design domain (ODD) exit. ODD essentially means that the vehicle can only operate in certain environments, small and defined, like specific motorways, or broad, like an entire trip. For instance, the ODD for Ford's BlueCruise is when driving in the predefined "blue zones", meaning that the user must take over when leaving them. While Level 3 autonomy has limited spread, some manufacturers are allowed to operate at this level, with Germany laying out the groundwork for the road to Level 4 autonomy @autoseu.

- *Level 4*: High Automation. _The vehicle performs all driving tasks under specific circumstances. Geofencing is required. Human override is still an option_.
The main difference between Level 3 and Level 4 is the fact that the vehicle can operate without human intervention in certain situations. This means that the vehicle can perform all driving tasks under specific circumstances, such as highway driving or in urban environments. This is the referred to as geofencing, where the vehicle can only operate in certain areas. As visualized in @fig:5levels, the steering wheel is now only outlined by dashes, meaning that the human driver is not in control of the vehicle at all times, but can take over if needed or required by the vehicle. 

There are currently no commercially available vehicles that reach this level of autonomy, but some manufacturers have developed and deployed what is essentially Level 4 systems. For instance, Waymo and Cruise have developed systems that can operate in certain areas, such as San Francisco and Phoenix. These systems are available to customer use, but they are developing vehicle any one can buy. Both Waymo and Cruise act as a taxi service, where users can request a ride through an app. As mentioned, however, these systems are only allowed to operate in specified, geofenced areas, and they are not allowed to operate outside of these areas. This is what is keeping these technologies at Level 4 instead of Level 5. They gather massive amounts of detailed data on the cities they operate in and train their systems on this data, meaning they are great in their respective areas, but they are not able to operate outside of them. Furthermore, Tesla unveiled their own robotaxis at the "We, Robot" event, featuring vehicles without steering wheels or pedals. While not commercially available either, they still present a glimpse of the future of Level 4 autonomy, potentially even Level 5. 

- *Level 5*: Full Automation. _The vehicle performs all driving tasks under all conditions. Zero human attention or interaction is required_.
Level 5 is the final level of the taxonomy presented by SAE. At this level, the ODD is unlimited, whereas every earlier level has been limited. At this level, both the DDT and DDT Fallback are fully handled by the vehicle. This means that the human user relinquishes the role of driver, and becomes a passenger, purely. The vehicle can operate in any environment, and the human user does not need to be aware of the environment at all. Not only does this mean that the vehicle can operate in any environment, but it also means that the vehicle can operate in conditions. If at any point during a trip, the system can't figure out how to navigate and for any reason requires human intervention, then it is not Level 5. This is neatly presented in the visualization in @fig:5levels, where there is no steering wheel in the image and the once-drive-now-passenger is sitting with a book, completely unaware of the environment.

This level is not available anywhere in the world yet, and it is still believed to be at least a decade away before being commercially available to consumers @level5_decade. This is most commonly the level depicted in futuristic sci-fi movies and games. Movies like "Minority Report" (2002), "Total Recall" (1990), and "Knight Rider" (1982) feature Level 5 #acrpl("AV") used for various tasks, such as autonomous driving, autonomous taxi services, and even fighting crime, respectively. Level 5 vehicles with personalities is a common trait in sci-fi, with the Delamain AI cab service from the game "Cyberpunk 2077" (2020) being a prime example. 

#align(center, [$ast$ #h(5mm) $ast$ #h(5mm) $ast$])

// Current methods for understanding real-world
// - sensors (lidar, radar, cameras)
// - sensor fusion (kalman filters, bayesian fusion), V2X (Car2X, DSRC)

To achieve Level 3 autonomy and higher, the vehicle must be able to understand the environment around it. This is done by various means and is done largely the same across the industry, with a few outliers. The most common way to understand the environment is through the use of sensors, such as cameras, radar, and lidar. Cameras are a fairly recent addition to the #acr("AV") technology stack. They are used to detect and understand the environment around the vehicle, such as detecting pedestrians, cyclists, and other vehicles @cams. This requires a fair amount of computing power and efficiency, as the vehicle must be able to process the data from the cameras in real-time. The technology required is relatively new, which is why it is only in recent years that cameras have become a mainstay in AVs, at least in the context of autonomous driving. Drivers have had rear-view and #acr("BEV") cameras for many years, however, these were only to assist said driver. Before cameras, it was common for vehicles to be equipped with radar and lidar systems. These systems offer a much higher resilience to adverse conditions, such as rain and fog, but they are also much more expensive. Lastly, ultrasonic sensors are used to detect objects in close proximity to the vehicle, such as when parking. These sensors are not used for understanding the broader environment, but they are used for detecting objects around the vehicle. A typical configuration os these sensors is shown in @fig:sensor_fusion#text("a", fill: theme.teal).

These systems are often used in conjunction with one another to create a single coherent picture of the environment. This is what is known as sensor fusion, where the deluge of data from the various sensors is combined. Not only does this create a coherent picture, but it also draws on the strengths and weaknesses of each of the aforementioned technologies. For example, a camera has great spatial resolution and little noise, but is poor at estimating velocity and distance where radar and lidar shine, respectively @cohen_2021. There are 3 types of sensor fusion classifications: Low-level, mid-level, and high-level. Low-level is considered early fusion, and the others late fusion.

#let fig1 = { image("../../figures/img/bg/early_fusion.png") }
#let fig2 = { image("../../figures/img/bg/late_fusion.png") }
#let fig3 = { image("../../figures/img/bg/sensors.png") }

#std-block(breakable: false)[
  #figure( 
    grid(
      columns: (1.39fr, 1fr),
      grid.cell(rowspan: 2, fig3),
      fig1, fig2,
      [#subfigure("(a)")], [#subfigure("(b)")]
    )
    ,
  caption: [(a) shows a typical sensor configuration, showcasing the large amount of different sensors on a single vehicle. (b) shows the difference between Early (top) and Late (bottom) fusion. Image source: (a) Yeong #etal @sensors, (b) Think Autonomous @sensor_fusion.]
) <fig:sensor_fusion>
]

Low-level fusion is considered early fusion, as it combines the raw data streams, i.e. camera pixels, lidar points, etc., from the sensors before any processing is done. While this method retains the most information possible from the sensors, it is also very computationally heavy @sensors. Mid-level fusion is considered late fusion, as it combines the intermediate representations. For instance, the vehicle's camera and radar might recognize a vehicle, then it recognizes those two representations as representing the same object. Finally, high-level fusing is also considered late fusion, as the combine the mid-level fusions with positional tracking. This is often through the use of probabilistic filters.

The most popular of these is the Kalman filter @kalman. It is a powerful recursive algorithm used in sensor fusion to estimate the state of a dynamic system by combining measurements from multiple, potentially noisy sensors over time. It consists of two main steps: the prediction step and the update step. In the prediction step, the Kalman filter uses a mathematical model of the system to predict the next state based on the current state and control inputs: 
$
  hat(x)_(k|k-1) = F_k  hat(x)_(k-1|k-1) + B_k  u_k
$
where $hat(x)$ is the state estimate, $F_k$ is the state transition model, $B_k$ is the control input model, and $u_k$ is the control input. Then, in the update step, the filter combines the predicted state with the new measurement to produce an updated estimate:
$
    hat(x)_(k|k) = hat(x)_(k|k-1) + K_k (z_k - H_k hat(x)_(k|k-1))
$
where $K_k$ is the Kalman gain, $z_k$ is the measurement, and $H_k$ is the observation model. By iteratively applying these steps, the Kalman filter fuses information from different sensors, accounting for their noise characteristics, to produce a statistically optimal estimate of the system's state. This sensor fusion is typically more accurate than relying on a single sensor, as it combines the strengths of each sensor type. This helps the vehicle understand its immediate environment, and the with help of technologies like #acr("V2X"), it can further understand the world around it. With all of these technologies in a vehicle's stack, it can make informed decisions and navigate dynamically complex environments effectively.

// - control systems (PID, MPC, LQR, reinforcement learning, gbp planner) (conjunction with path planning, robot control problem)
// - decision making (rule-based, behavioural cloning, imitation learning, reinforcement learning)

With this robust and detailed understanding of the world, many different control methodologies have been developed to control vehicles and robots, designated as the control problem. The most common and simple method is the Proportional-Integral-Derivative (PID) controller. Briefly, PID controller always try to minimize the error between the measured state and a desired. This error is then subject to three different operations: proportional, integral, and derivative, each multiplied by some scalar. Each property influences things like speed, overshoot, and stability. A more complex method is the Model Predictive Control (MPC) method. This method uses a model of the system to predict the future states and optimize the control inputs over a finite time horizon. It is particularly useful for systems with constraints, such steering angle, acceleration, and velocity. MPC is often used in conjunction with path planning, where the vehicle must follow a specific path while avoiding obstacles. 

While path-planning is covered in @c3:related-work, how AVs achieve this will briefly be covered, for with this robust representation of the environment and the control methodologies, the vehicle can make informed decisions and navigate dynamically complex environments effectively. Today, this is largely done with the use of #acr("DL"), but more manual methods exist. These methods are less generalizable than DL ones, but are robust in their own ways. One such method is built up around the concept of a state machine, known as rule-based methods. These are scenario specific rules that are applied at predefined objectives, such as slowing down automatically when seeing a red light. For DL approaches, data-driven approaches are common. Behavioural cloning and imitation learning are common when you have massive amounts of driving data. These methods are trained on the data, and learn to mimic the behaviour of the human driver. This is done by training a #acr("NN") to predict the control inputs based on the sensor data. #acr("DL") approaches have witnessed massive growth in the last decade, and is now the most common methodology for AVs.


== Deep Learning <c2s2.2:dl>

The groundwork for the modern #acr("AI") we see today, has been under development since the 1940's. In 1943, Warren McCulloch and Walter Pitts proposed the artificial neuron @neuron. Their goal was to create a model that acted like the human brain, in that brains are made up of constant firing neurons. They proposed #acrpl("NN") can be modelled using a logical calculus based on the “all-or-none” firing principle of neurons. They demonstrate that neural activations correspond to logical statements. This provides a systematic way to analyse and predict neural behaviour, but also lays the theoretical groundwork for later advances in computational neuroscience and AI. Already in 1951, Minsky and Edmonds presented the Stochastic Neural Analog Reinforcement Calculator (SNARC), which consisted of 40 artificial neurons @snarc. As will become the norm, this network was trained by adjusting the strengths of the connections between neurons based on the outcomes of the previous trials.

Another concept introduced early, was what would come to be known as reinforcement learning. Arthur Samuel, often called the father of machine learning, introduced the Samuel Checkers-Playing program in 1952, which learned the game of checkers via self-learning, improving its skills over time by playing many more games than a human ever could @ibm_checkers. The term AI was coined in 1956, but it was not until 1958 Frank Rosenblatt developed the first #acr("ANN"), which he named the perceptron, which was a single-layer #acr("NN") that could learn to classify patterns @Rosenblatt1958-bm. Building on the early breakthroughs, the following decades saw an expansion of ideas that sought to overcome the limitations of single-layer #acrpl("NN"). Researchers began experimenting with multi-layered architectures and early variants of backpropagation @ai_timeline. In general, this period laid important groundwork for the following decades in the research and development of #acr("AI") and, subsequently, #acr("DL").

One of the most important breakthroughs came in the form of the rediscovery and refinement of backpropagation @backprop. This sparked the renaissance in #acr("NN") research, as interest in learning from data was renewed. This breakthrough enabled the development of #acr("DL") architectures, which could learn complex patterns and representations from large datasets. The following decades saw the integration of statistical learning methods with #acr("NN") architectures that further enhanced their accuracy and robustness. What is considered the first point of contact with AI, came in 2012 with the release of AlexNet @alexnet. This model was capable of classifying images with a high degree of accuracy, paving the way for modern learning algorithms and #acr("DL") architectures. 

Looking at the highlights of the previous paragraphs, the most important take-aways are as follows. McCulloch and Pitts stated that "At any instant a neuron has some threshold, which excitation must exceed to initiate an impulse." which, as will be presented, is exactly how modern machines are taught to think and learn. SNARC used a method of adjusting the strengths of the connections between neurons based on the outcomes of the previous trials. Samuel's Checkers-Playing program used a method of self-learning, which is the basis of modern reinforcement learning. Rosenblatt's perceptron was the first ANN, introducing the concept of layers in a NN, later expanded to the modern #acr("MLP").

// Fundamentals
// - MLP, backpropagation, gradient descent
// - Activation functions, optimizers, loss functions, regularization, scheduler, logits

The #acr("MLP") is the most common and simple type of #acr("NN") used in DL. It consists of an input layer, one or more hidden layers, and an output layer. Each layer consists of a number of neurons, which are connected to the neurons in the previous and next layers. The connections between the neurons are weighted, and these weights are adjusted during training to minimize the error between the predicted output and the actual output. The training process is done using backpropagation, which is a method for calculating the gradients of the weights with respect to the error. This is done by applying the chain rule of calculus to calculate the gradients of each weight in the network. 

This process will now be broken down into its components, with the intent of creating a clear understanding of how #acrpl("MLP") work, and how they are trained. This will act as a springboard for understanding the more complex architectures and methods used in this work.

#let fig1 = { image("../../figures/img/bg/mlp.svg") }

#std-block(breakable: false)[
  #figure(
    fig1,
    caption: [Multi-Layer Perceptron (MLP).]
  ) <fig:mlp>
]

#acrpl("MLP") are composed of a number of layers, each consisting of a number of neurons. Each neuron is connected to the neurons in the previous and next layers. The connections between the neurons are weighted (as shown by the different opacities of the connections in @fig:mlp). MLPs are often what is called "fully connected", meaning that each neuron in a layer is connected to every neuron in the next layer. This is done to allow for the maximum amount of information to be passed between the layers. 

To start from the smallest possible unit, the neuron, it is important to understand how they work. During a forward pass through a NN (giving it some input feature), the value of each neuron is calculated as follows:
$
  z = bold(w)^T bold(x) + b = sum_i bold(w)_i x_i + b
$ <eq:neuron>
where $bold(w)$ is the weight vector, $bold(x)$ is the input vector, and $b$ is the bias term. The bias term is a constant that is added to the weighted sum of the inputs. This allows the neuron to learn a threshold value, which is important for learning complex patterns. The output of the neuron $a$ is then calculated using an activation function $a = phi(z)$, which introduces non-linearity into the model. Three of the most common activation functions $phi$ are shown below:

#let fig1 = { image("../../figures/img/bg/sigmoid.png") }
#let fig2 = { image("../../figures/img/bg/tanh.png") }
#let fig3 = { image("../../figures/img/bg/relu.png") }
#let col = rgb("#06DACA")
#let big_math(s: 14pt, body) = {
  show math.text: set text(size: s)
  body
}

#std-block(breakable: false)[

  #grid(
    align: (center, center, center),
    columns: (1fr, 1fr, 1fr),
    row-gutter: 5mm,
    fig1, fig2, fig3,
    [#big_math[$sigma(z) = 1/(1+e^(-z))$]], [#big_math[$tanh(z) = (e^z - e^(-z))/(e^z + e^(-z))$]], [#big_math[$"ReLU"(z) = max(0, z)$]]

  )
]

The first of the three is the sigmoid function. It maps the input to a value between $0$ and $1$, making it particularly useful for binary classification tasks. tanh scales the output to a value between $-1$ and $1$, often leading to faster convergence. The last one, ReLU, is the most common activation function used in #acr("DL") today. It is defined as $"ReLU"(z) = max(0, z)$, meaning that it outputs the input directly if it is positive, and $0$ otherwise. This function is computationally efficient and helps mitigate the vanishing gradient problem, which can occur with sigmoid and tanh functions. The vanishing gradient problem occurs when the gradients of the weights become very small, making it difficult for the model to learn.

Another important activation function is the softmax function, which is used in the output layer of a #acr("MLP") for multi-class classification tasks. It converts the raw output logits into probabilities by exponentiating each logit and normalizing them:
$
  "softmax"(z_i) = e^(z_i) / (sum_j e^(z_j))
$ <eq:softmax>
which ensures that the sum of the probabilities is equal to $1$. Now, here is an example of a forward pass through the MLP. First, the input features $bold(x)$ is passed through the hidden layer:
$
  bold(z)^((1)) = bold(W)^((1))bold(x) + b^((1)) \
  bold(a)^((1)) = phi(bold(z)^((1)))
$ <eq:mlp1>
where $bold(W)^((1))$ is the weight matrix for the first layer, $bold(z)^((1))$ is the weighted sum of the inputs, and $bold(a)^((1))$ is the output of the first layer. The output of the first layer is then passed through the output layer:
$
  bold(z)^((2)) = bold(W)^((2))bold(a)^((1)) + b^((2)) \
  hat(y) = "softmax"(bold(z)^((2)))
$ <eq:mlp2>
where, in this example, the output $hat(y)$ is put through a softmax activation to get probabilities on the output. Depending on the task of the NN, different loss functions are used. Loss functions are used to measure the difference between the predicted output and the actual output. For the task of regression (predicting a continuous value), the most common loss function is the Mean Squared Error (MSE) loss, which is defined as:
$
  L(bold(hat(y)), bold(y)) = 1/N sum_i (hat(y)_i - y_i)^2
$ <eq:mse>
where $N$ is the number of samples, $hat(y)$ is the predicted output, and $y$ is the actual output. For classification tasks, the most common loss function is the Cross-Entropy loss, which is defined as:
$
  L(bold(hat(y)), bold(y)) = -sum_i y_i log(hat(y)_i)
$ <eq:ce>
which calculates the dissimilarity between the predicted distribution $bold(hat(y))$ and actual distributions $bold(y)$. The binary version, called #acr("BCE"), is also commonly used for binary classification tasks, and is defined as:
$
  L(bold(hat(y)), bold(y)) = -sum_i (y_i log(hat(y)_i) + (1 - y_i) log(1 - hat(y)_i))
$ <eq:bce_eq>

#let fig1 = { image("../../figures/img/bg/gradient_descent.png") }
#let fig = {std-block(breakable: false)[#figure(fig1, caption: [Gradient descent. \ Image source: WikiPedia @grad]) <fig:gradient_descent>]}

#let c = [where $y_i$ is the actual label, and $hat(y)_i$ is the predicted probability of the positive class. Once the forward pass is complete and the loss $L$ is calculated using a function like MSE or Cross-Entropy, the goal of training the neural network is to minimize this loss. This done through what is called gradient descent. Gradient descent is a method for mathematical optimization, meaning by doing certain operations, a network's parameters can reach minima, where the error, or loss, is the smallest it can be. This is done by calculating the gradients of the loss with respect to the parameters, and updating the parameters in the opposite direction of the gradients. The goal is to reach what is called a local minima, where the loss is minimized. This minima symbolizes the best (smallest) loss that a network can achieve depending on its weights and biases.]

#wrap-content(fig, c, align: right, columns: (2fr, 1fr))

#h(4mm) Formally, this is known as backpropagation, which is a method for calculating the gradients of the loss with respect to the parameters. This is done by applying the chain rule of calculus to calculate the gradients of each weight in the network. So, the way to find the gradient for a weight $w_(i j)$ connecting neuron $i$ to neuron $j$ in the next layer is to calculate the partial derivative of the loss with respect to that weight:
$
  (partial L) / (partial w_(i j)) = (partial L)/(partial z_j) (partial z_j)/(partial w_(i j))
$ <eq:grad>
where
$
  z_j = sum_i w_(i j) x_i + b_j #h(6mm) => #h(6mm) (partial z_j)/(partial w_(i j)) = x_i
$ <eq:grad2>
meaning the gradient is 
$
  (partial L)/(partial w_(i j)) = delta_j x_i
$ <eq:grad3>
where $delta_j$ is the error term for neuron $j$, calculated using one of the aforementioned loss functions. With this, it is now time to optimize the weights. This is done by using one of the optimization algorithms available. Two of the most commonly used optimizers are Stochastic Gradient Descent (SGD) and Adam. SGD is a simple and effective optimization algorithm that updates the weights using the gradients calculated during backpropagation. The update rule for SGD is:
$
  w_(i j)^((t+1)) = w_(i j)^((t)) - eta (partial L)/(partial w_(i j))
$
where $eta$ is the learning rate, which controls the step size of the update. The learning rate is often a very small value, as to help NNs converge to a local minima slowly. A fair bit more involved and extremely influential is the Adam optimizer. Adam means Adaptive Moment Estimation, and is an adaptive learning rate optimization algorithm. It combines the advantages of two other extensions of SGD: momentum and RMSProp. It maintains two moving averages for each parameter: one for the gradients (first moment) and one for the squared gradients (second moment). It also includes a bias-correction mechanism to counteract the initialization bias of the first moment estimates. The update rules for Adam at time $t$ are as follow:
#enum(
  [Compute the gradients: $g_t = (partial L)/(partial w_t)$],
  [Update the biased first moment estimate: $m_t = beta_1 m_(t-1) + (1 - beta_1) g_t$],
  [Update the biased second moment estimate: $v_t = beta_2 v_(t-1) + (1 - beta_2) g_t^2$],
  [Bias-correct the moment estimates: $hat(m)_t = (m_t)/(1 - beta_1^t)$, $hat(v)_t = (v_t)/(1 - beta_2^t)$],
  [Update the parameters: $w^((t+1)) = w^((t)) - eta (hat(m)_t)/(sqrt(hat(v)_t) + epsilon)$]
)
where $beta_1$ and $beta_2$ are the decay rates for the first and second moment estimates, respectively. These are typically very close to 1, commonly set to 0.9 and 0.999. $epsilon$ is a small constant added to prevent division by zero, and is typically set to $10^(-8)$. Both $m_0$ and $v_0$ are initialized as $0$. While a constant learning rate $eta$ is often good enough, too small of a constant $eta$ may result in the models being stuck in the closest local minima, which is likely not the best. A higher $eta$ is not exactly the way to combat this, as the training may become highly unstable, if the changes in weights are too large. This is where learning rate scheduling comes in. Learning rate scheduling is a technique used to adjust the learning rate during training. There are several different types of learning rate schedules, such as step decay, exponential decay, and cosine annealing. 

Step decay is the simplest, where the learning rate is reduced by a factor every few epochs. Exponential decay is largely the same, but it is constantly decreasing the learning rate each epoch. First, the step decay defines the learning rate as:
$
  eta^((t)) = eta_0 dot gamma^floor(t / "step-size")
$ <eq:step>
where $t$ is the epoch, $eta_0$ is the initial learning rate, and $gamma$ is the decay factor. $floor(dot)$ denotes the floor function. Step decay is simple and can help a model settle into a local minima, but it is not very flexible. The non-smooth nature of the floor function used in the step decay can lead to abrupt changes in the learning rate, which can cause instability. Furthermore, it requires careful tuning of the step size and decay factor. Exponential decay combat some of these disadvantages by using a continuous decay function. The learning rate is defined as:
$
  eta^((t)) = eta_0 dot e^(-k dot t)
$ <eq:exp>
where $k$ is the decay constant. This allows for a smoother and more gradual decrease in the learning rate, which can help the model converge more effectively. However, it still requires careful tuning of the decay constant $k$. It is, however, still a smooth uniform decay, which means it doesn't help the model potentially escape poor local minima. This is where cosine annealing comes in. Cosine annealing is a more advanced learning rate schedule that uses a cosine function to adjust the learning rate. The learning rate is defined as:
$
  eta^((t)) = eta_min + 1/2 (eta_max - eta_min) (1 + cos((pi t)/ T))
$ <eq:cosine_annealing> 
where $eta_min$ is the minimum learning rate, $eta_max$ is the maximum learning rate, and $T$ is the total number of epochs for the annealing cycle. This cosine function provides a periodic transition between high and low learning rates. When coupled with warm restarts (occasionally setting $eta^((t)) = eta_max$), this scheduler helps the optimizer escape local minima and thereby increase performance. The downsides are that it is more complex than the other schedulers, and it requires careful tuning of more parameters than both step and exponential decay.

Finally, some other common concepts w.r.t training are these:
- Regularization: A technique used to prevent overfitting by adding a penalty term to the loss function. The most common form of regularization is L2 regularization, which adds a term to the loss function that is proportional to the square of the weights. This encourages the model to learn smaller weights, which can help prevent overfitting. L2, or weight decay, is defined as: $L_"total" = L_"data" + lambda ||w||^2_2$.  
- Dropout: A technique used to prevent overfitting by randomly dropping out a fraction of the neurons during training. This forces the model to learn more robust features and prevents it from relying too heavily on any one neuron. 
- Batch Normalization: A technique used to normalize the inputs to each layer in the network. First the batch mean and variance are found:
$
  mu_B = 1/m sum_(i=1)^m x_i, \
  sigma_B^2 = 1/m sum_(i=1)^m (x_i - mu_B)^2
$ <eq:batch_norm>
where $m$ is the batch size. Then, the inputs are normalized:
$
  hat(x)_i = (x_i - mu_B) / sqrt(sigma_B^2 + epsilon)
$ <eq:batch_norm_normalized>
where $epsilon$ is a small constant added to prevent division by zero. Finally, the normalized inputs are scaled and shifted:
$
  y_i = gamma hat(x)_i + beta
$ <eq:batch_norm_scaled>
where $gamma$ and $beta$ are learnable parameters that allow the network to scale and shift the normalized output. 

In summary, the architecture and training process of the MLP has been presented in detail. By starting with creating an understanding of how each neuron worked, to how they are connected, and how the entire MLP is trained, a solid understanding of the MLP has been created, and generally how to train NNs. The forward and backward propagation was presented, as well as the loss functions and optimizers. The most common activation functions were presented, as well as the most common learning rate schedulers. With all of these methods, it is possible to train our MLP at a task we desire. However, the MLP is too simple of an architecture to learn any really complex tasks. This is where the more complex architectures and methods come in. 

#align(center, [$ast$ #h(5mm) $ast$ #h(5mm) $ast$])

Before moving on to the complexities introduced in the subfield of computer vision, it is important to note that DL consists of three main learning paradigms. The first, and most common, of these is supervised learning. Supervised learning is concerned with training models using labelled data. This means that the model is trained on a dataset where the input features are paired with the corresponding output labels, i.e. the dataset consists of pairs $(x,y)$ where $x$ is the input features and $y$ is the corresponding labels, often referred to as the ground truth labels. With each of these pairings, the training process involves minimizing a loss function, as described earlier, between the input features $x$ and the ground truth labels $y$. Therefore, this learning paradigm is often used for image classification tasks, speech recognition, and natural language processing tasks. In other words, supervised learning is for when you know how you want the outcome to look by coming as close to the ground truth as possible. 

Alternatively, you might not know exactly how your output should look. This is where unsupervised learning comes in. Unsupervised learning is concerned with training models using unlabelled data. This means that the model is trained on a dataset where the input features are not paired with any output labels. The goal of unsupervised learning is to learn the underlying structure of the data, such as clustering similar data points together or reducing the dimensionality of the data. This learning paradigm is often used for tasks such as clustering, anomaly detection, and dimensionality reduction. Furthermore, this paradigm is typically used when training generative models. Generative models are models that learn to generate new data points that are similar to the training data. This is done by learning the underlying distribution of the data, and then sampling from that distribution to generate new data points. This task is most often seen in image generators, such as #acrpl("GAN").

Finally, #acr("RL") is a learning paradigm where a model learns through interactions with the environment. The model learns to take actions in an environment to maximize a reward signal. This is done by learning a policy, which is a mapping from states to actions, and a value function, which is a mapping from states to expected rewards. The goal of reinforcement learning is to learn a policy that maximizes the expected cumulative reward over time. This learning paradigm is often used for tasks such as game playing, robotics, and autonomous driving. Popular examples include AlphaGo #citation_needed beating the world's greatest Go player, and OpenAI Five #citation_needed beating the world champions in Dota 2.

These three paradigms broadly cover the methods used to train different models for different tasks. Each offers their strengths and weaknesses. Labelling data can be a time-consuming task, but create faster more targeted training, where unlabelled data and environmental interaction may reach beyond the performance initials goals. Each paradigm has their own objectives when training, from minimizing some prediction error to discovering hidden patterns to maximizing some cumulative reward. This also means that each paradigm lend themselves to achieve specific goals, such as classification, anomaly detection, and decision making, respectively. Computationally, they also offer different complexities: supervised learning can vary a lot depending on the task, but is typically moderately complex; unsupervised learning is often more complex, as the model must learn the underlying structure of the data; and reinforcement learning is often the most complex, as it requires the model to learn through iterative interactions with the environment.

These paradigms form the backbone of machine learning techniques used across various domains, including computer vision. Specifically, supervised learning has driven significant advancements in tasks like image classification, where labelled datasets guide models to recognize intricate visual patterns. One landmark achievement highlighting the potential of supervised methods in computer vision is AlexNet, which is widely regarded as the starting point of modern artificial intelligence.

// Types: Reinforcement Learning, Supervised Learning, Unsupervised Learning, transfer learning, self-supervised learning
// point of first contact with AI as lead-in to CNNs
// Recurrent Neural Networks and LSTMs (sequential decision-making), 
// Graph NN (road topology, scene graph modelling)

=== Computer Vision <c2s2.2.1:cv>
AlexNet is widely considered to be the first point of contact with what we classify as AI today @alexnet. It was an image classification model that was trained on the ImageNet dataset, and was capable of classifying images with a high degree of accuracy. This was the first NN to clearly pass other machine learning methods, such as kernel regression and support vector machines, in image classification tasks. This was a major breakthrough in the field of computer vision, opening the field to more researchers by proving the capabilities if NNs. Since its release, the interest in computer vision tasks has exploded, and is thus deployed in a wide range of applications, such as image classification, object detection, image segmentation, pose estimation, and image generation. 



#let fig1 = { image("../../figures/img/bg/classification.png") }
#let fig2 = { image("../../figures/img/bg/object_detection.png") }
#let fig3 = { image("../../figures/img/bg/semantic_segmentation.png") }
#let fig4 = { image("../../figures/img/bg/instance_segmentation.png") }
#let fig5 = { image("../../figures/img/bg/pose_estimation.png") }
#let fig6 = { image("../../figures/img/bg/image_generation.png") }

#std-block(breakable: false)[
  #figure(
    grid(columns: (1fr, 1fr, 1fr),
    fig1, fig2, fig3, 
    [#subfigure("(a)")],[#subfigure("(b)")],[#subfigure("(c)")],
    fig4, fig5, fig6,
    [#subfigure("(d)")],[#subfigure("(e)")],[#subfigure("(f)")],
    
    ),
    caption: [(a) Image classification. (b) Object detection. (c) Semantic segmentation. (d) Instance segmentation. (e) Pose estimation. (f) Image generation. Image source: (a-e) viso.ai @cv_tasks (f) Altered with OpenAI 4o @4o-image.]
  ) <fig:cv_tasks>
]

#h(4mm) The image classification task is the most simple of the computer vision tasks. It is concerned with classifying an image into one of a number of classes. This task, which was AlexNet's main task, is shown in @fig:cv_tasks#subfigure("a"). As shown, it is concerned with labelling the entire image as one class. It should be noted "class" means the name of a group of similar things, like the example shows the class of the image is "tiger". This ideas of classes, becomes important when moving on to the other CV tasks. The next task is object detection, which is concerned with detecting and localizing objects in an image. This task is shown in @fig:cv_tasks#subfigure("b"). It is concerned with not only classifying the image, but also localizing the objects in the image. This is done by drawing bounding boxes around the objects in the image, and classifying them. Note that there may be multiple instances of the same objects in the image, each requiring their own bounding box for correct classification.

The next task is semantic segmentation, which is concerned with classifying each pixel in the image. This task is shown in @fig:cv_tasks#subfigure("c"). It is concerned with classifying each pixel in the image into one of a number of classes. For this task, the model needs to be trained to not just identify the objects in the image, but also to identify the boundaries of the objects. This is done by creating a class label or class mask in the training data. A class mask is an image where each pixel is assigned a class label, and as mentioned prior, this class mask is the ground truth for said image. Closely related is the task of instance segmentation, which is concerned with classifying each pixel in the image, but also differentiating between different instances of the same object. This task is shown in @fig:cv_tasks#subfigure("d"). The main difference between these two segmentation tasks, is the fact that semantic segmentation does not differentiate between different instances of the same object. This means that if there are two objects of the same class in the image, they will be classified as the same object. This is not the case for instance segmentation, where each instance of the same object is classified as a different object. 

The next task is pose estimation, which is concerned with estimating the pose of an object in the image. It can be split into two separate tasks of its own: estimation of the 3D position of an object within an image and estimating the pose of the isolated object. The latter of these is shown in @fig:cv_tasks#subfigure("e"), where a rigging skeleton is drawn over the object to indicate its pose. This is done by estimating the position of the joints in the image, and drawing lines between them. The former task is concerned with estimating the 3D position of an object in the image, often achieved through a cascade of methods, such as 3D reconstruction, depth estimation, and multi-view stereo techniques. 

The final task is image generation, which is concerned with generating new images from a given input. This task is shown in @fig:cv_tasks#subfigure("f"), where the image of the tiger has been altered to mimic a specific art style. This is a very common usage of image generators, but they can also create images from the ground up. In GANs, this is done by training the model on a dataset of images, and then generating new images that are similar to the training data. Other methods like diffusion iterate of noisy images, slowly refining them to create a new image based on some description. This is done by training the model to learn the underlying distribution of the data, and then sampling from that distribution to generate new images.

With these tasks in place, the methods for which either is achieved will now be presented. AlexNet consisted mainly of convolutional layers, with the occasional pooling layer. Convolution is a very important mechanic within the field of computer vision, as it allows for models to gain an understanding of the features within an image. 


#let fig1 = { image("../../figures/img/bg/alexnet.svg") }
#let fig2 = { image("../../figures/img/bg/convolution.png") }
#let fig3 = { image("../../figures/img/bg/pooling.png") }

#std-block(breakable: false)[
  #figure(
    grid(
      columns: (1fr, 1fr),
      align: horizon,
      grid.cell(colspan: 2, fig1), grid.cell(colspan: 2, { subfigure("(a)") }),
      fig2, fig3,
      subfigure("(b)"), subfigure("(c)")
    ),
    caption: [AlexNet architecture.]
  ) <fig:alexnet>
]

Convolution is achieved by convolving a kernel over the input image. This is done by sliding the kernel over the image, and at each position, calculating the dot product between the kernel and the image. This is done by multiplying each element in the kernel with the corresponding element in the image, and then summing the results. The result of this operation is a new image, called a feature map. This map is typically smaller than the original image, as the kernel only stays within the bounds of the image itself. This can be bypassed by padding the input image with zeros, which allows the kernel to convolve along the very outer edge of the image. For input image often containing three channels (RGB), the kernel is convolved with each channel separately, or, rather, the kernel is three dimensional. This outputs one channel, which is why many different kernels are used. As seen in @fig:alexnet#subfigure("a"), the first layer of AlexNet after the input layer consists of 96 channel, with the next being 256 channels. 

Other than padding, other factors come into the usage of convolution kernels. The stride is the number of pixels the kernel is moved at each step. A stride of $1$ means that the kernel is moved one pixel at a time, while a stride of $2$ means that the kernel is moved two pixels at a time. This can be used to reduce the size of the feature map, as the kernel will skip some pixels in the image. This is also seen in the figure, where the kernel is a large $11 times 11$ kernel with stride $4$. This results in the output being significantly smaller than the input image. Next, the kernel size is the size of the kernel itself, and is typically a small odd number, such as $3$, $5$, or $7$. The kernel size is important, as it determines the size of the receptive field of the model. The receptive field is the area of the input image that the kernel is able to see at any given time. A larger kernel size means a larger receptive field, which allows the model to learn more complex features. However, a larger kernel size also means more parameters to train, which can lead to overfitting. 

Padding, stride, and kernel size are also relevant to the other operation incorporated into AlexNet: pooling. Pooling is a downsampling operation that reduces the size of the feature map. This is done by taking the maximum or average value of a small region in the feature map, and using that value as the new value for that region. This is done by sliding a pooling kernel over the feature map, and at each position, calculating the maximum or average value of the region covered by the kernel. The result of this operation is a new feature map, which is smaller than the original feature map. 

These concepts of stride and pooling are illustrated in @fig:alexnet#subfigure("c"). The size of the pooling kernel is the often the same as the stride, meaning the kernel does not encompass any overlapping pixels. As shown with the right-hand side example, the stride is $1$, which results in overlapping regions. This is a rather undesired effect, so a stride equal to the kernel size is often used. This is shown in the left-hand side example, where the stride is $2$, and the pooling kernel is $2 times 2$. This results in a non-overlapping pooling operation, which is often desired. The most common pooling operations are max pooling and average pooling. Max pooling takes the maximum value of the region covered by the kernel, while average pooling takes the average value of the region covered by the kernel. Examples of both are shown in @fig:alexnet#subfigure("c") on the left-hand side of the input.

Finally, a few more components of the AlexNet architecture are worth mentioning. After each convolutional layer, a ReLU activation function is applied. This is done to introduce non-linearity into the model, which allows it to learn more complex features. The final layer of AlexNet is a fully connected layer, which is used to classify the image into one of the classes. This is done by flattening the feature map into a vector, and then passing it through a series of fully connected layers. The final output is then passed through a softmax activation function, which converts the output into probabilities for each class. AlexNet utilizes dropout for regularization, as to prevent overfitting. This is done by randomly dropping out a fraction of the neurons during training, which forces the model to learn more robust features. Furthermore, to prevent overfitting, during training, the input images are put through augmentation. This will be covered in greater detail in @c3:data_augmentation. Finally, AlexNet also uses Local Response Normalization (LRN) layers. It works by normalizing the activations of neurons in a local region across the channel dimension.

Moving beyond the convolutional NN (CNN) architecture that is AlexNet, many other methods and architectures exist to help machines understand images. Closely related by the fact that it is also considered a CNN, is the U-Net  architecture, which is widely used for image segmentation tasks #citation_needed. U-Net is a fully convolutional network that consists of an encoder and a decoder. The encoder is responsible for downsampling the input image, while the decoder is responsible for upsampling the feature map back to the original size. This is done by using skip connections between the encoder and decoder, which allows the model to learn both low-level and high-level features. The upsampling, opposed to downsampling/pooling, is done by using transposed convolutions, which are the reverse of normal convolutions. Whereas normal convolution reduce patches to singular values, transposed convolution does the opposite; it take a singular value and expands it to a patch using a kernel. This typically doubles the height and width of the feature map. Convolutions in general see heavy usage in CV tasks, as the kernel they use are learnable, meaning they are affected by backpropagation, meaning they can learn.

#align(center, [$ast$ #h(5mm) $ast$ #h(5mm) $ast$])

In 2020, ChatGPT hit the ground running, exploding into the public conscience and making AI a mainstream tool that everyone suddenly knew about #citation_needed. This leap in Natural Language Processing (NLP) was made possible by the introduction of the Transformer architecture, which was introduced in 2017. In their landmark paper "Attention is All You Need", Vaswani #etal #citation_needed introduced the Transformer architecture, which revolutionized the field of NLP. The transformer architecture is based on the self-attention mechanism, which allows the model to weigh the importance of different words in a sentence when making predictions. To understand self-attention, imagine reading a sentence. When you read a word, you don't just look at the word itself, but also at the words around it. In the sentence "The cat chased the mouse because it was fast", the word "it" refers to "the mouse". Self-attention allows a model to do the same computationally. The self-attention mechanism works by calculating how relevant a word in a sentence is to every other word in the same sentence. This allows the model to look at other parts of the input sequence and determine which parts are most important for predicting the next word. This mechanism is heavily utilized in the transformer architecture. Transformers can process all words in a sequence in parallel. This is a major advantage over recurrent NNs (RNNs), which process words sequentially. They typically consist of an encoder and a decoder, where the encoder processes the input sequence and the decoder generates the output sequence. Both use multiple layers stacked with self-attention and feed-forward networks. Thus, transformers helped technologies, like the Generative Pre-trained Transformers (GPTs), gain an incredible understanding of human language and explode into the public conscience.

Inspired by the self-attention mechanism and transformers, Dosovitskiy #etal @vit_og introduced the #acr("ViT") architecture in 2021. This extended the capabilities of transformers into the field of computer vision. To make transformers applicable to images, the input image is divided into fixed-size patches, which are then flattened and linearly embedded into a sequence of tokens. This is very much akin to the way words in a sentence are processed in NLP. This allows the transformer to treat visual data in a similar way to text data, enabling it to learn complex relationships between different parts of the image through the self-attention mechanism. The ViT architecture consists of a series of transformer blocks, each containing a multi-head self-attention layer and a feed-forward network. The output of the final transformer block is then passed through a classification head to produce the final output. So, the vision architecture typically consists of three main steps:

#let fig1 = { image("../../figures/img/bg/vit.png") }
#let fig = {std-block(breakable: false)[#figure(fig1, caption: [Vision Transformer architecture. \ Image source: Dosovitskiy #etal #citation_needed]) <fig:vit>]}

#let c = [*Patch embedding*: An input image is divided into equally-sized non-overlapping patches, typically 16x16 pixels each. These patches are flattened into vectors and transformed into embeddings through a #text("learned linear projection", fill: rgb(252, 225, 224), stroke: 0.1pt + rgb(252, 225, 224).darken(40%)), producing fixed-dimensional patch embeddings. *Transformer encoder*: The embeddings go through multiple stacked layers of #text("transformer encoder blocks", fill: rgb(229, 229, 229), stroke: 0.1pt + rgb(229, 229, 229).darken(40%)), each comprising two sub-layers: a #text("multi-headed self-attention mechanism", fill: rgb(199, 232, 172).darken(20%)) and a #text("feed-forward neural network", fill: rgb(193, 228, 247), stroke: 0.1pt + rgb(193, 228, 247).darken(10%)). Multi-headed self-attention computes attention scores to weigh the relevance of each patch relative to all others simultaneously. To retain spatial information, #text("positional embeddings", fill: rgb(209, 188, 210)) are added to the patch embeddings before being input into the transformer layers. *Classification head*: The processed embeddings from the transformer layers include a special classification token (`[class]`), prepended to the sequence of patch embeddings. The `[class]` embedding captures a global representation of the image. This global embedding is then passed through a #text("fully-connected neural network", fill: rgb(255, 210, 164)) to produce class predictions for image classification tasks.]

#wrap-content(fig, c, align: right, columns: (1fr, 2fr))

#h(4mm) Closely related to the ViT is the Swin Transformer, introduced by Liu #etal #citation_needed in 2021. The Swin Transformer is a hierarchical transformer that uses a shifted windowing scheme to reduce the computational cost of self-attention. This is done by dividing the input image into non-overlapping windows, and then applying self-attention within each window. These windows are merged in the deeper layers of the network, achieving great highly accurate segmentation masks. 


#let fig1 = { image("../../figures/img/bg/loss_plot.png") }
#let fig = {std-block(breakable: false)[#figure(fig1, caption: [Training and evaluation loss plot. ]) <fig:loss_bg>]}
#let c = [Now, a term I have not explained yet is "overfitting". Overfitting is a common problem in machine learning, where the model learns the training data too well, and is unable to generalize to new data. This phenomenon is seen in the training and evaluation (or test or validation) graphs of a model's training process. Early in the training process, the model's performance on both the training and test sets improve as the model learns the task. However, as training continues, the model's performance on the training set continues to improve, while the performance on the test set starts to degrade. This is a sign that the model is overfitting to the training data. This is visualized in @fig:loss_bg, where the training loss continues to decrease, while the validation loss starts to increase after training iteration 75.  ]

#wrap-content(fig, c, align: right, columns: (1fr, 1.5fr))

#h(4mm) To combat overfitting, several techniques are used. As presented already, dropout is commonly used to combat overfitting. This is done by randomly dropping out a fraction of the neurons during training, which forces the model to learn more robust features. Another common technique is early stopping, where the training process is stopped when the performance on the validation set starts to degrade. This is done by monitoring the validation loss during training, and stopping the training process when the validation loss starts to increase too significantly. As shown, the validation loss comes across as noisy, so before employing early stopping, it is important to be certain that the validation loss is getting consistently worse. Alternatives to early stopping is simply by using a checkpoint system, where the weights of the model, or its current state, are saved at regular intervals. Regularization is also a common method to combat overfitting. This is done by adding a penalty term to the loss function, which encourages the model to learn smaller weights. The most common form of regularization is L2 regularization, which adds a term to the loss function that is proportional to the square of the weights. This encourages the model to learn smaller weights, which can help prevent overfitting. L1 regularization is also used, which adds a term to the loss function that is proportional to the absolute value of the weights. 

Finally, data augmentation comes in many shapes and sizes, literally. Data augmentation is a technique used to artificially increase the size of the training dataset by applying various transformations to the input data. A selection of these is shown in @fig:data_augmentation. These data augmentation techniques are used to create new training samples by applying various transformations to the original image. This is done to increase the diversity of the training data, which can help improve the performance of the model. The shown examples are very common in computer vision tasks. Finally, a technique called cross-validation is often used. This methods splits the dataset into multiple subsets, or folds. During training, one of the folds is used for validation, while the others are used for training. This is done to ensure that the model is not overfitting to a specific subset of the data. 

#let fig1 = { image("../../figures/img/bg/tiger_square.png") }
#let fig2 = { image("../../figures/img/bg/cropped_image.png") }
#let fig3 = { image("../../figures/img/bg/rotated_image.png") }
#let fig4 = { image("../../figures/img/bg/flipped_image.png") }
#let fig5 = { image("../../figures/img/bg/central_cropped_image.png") }
#let fig6 = { image("../../figures/img/bg/blurred_image.png") }
#let fig7 = { image("../../figures/img/bg/saturation_adjusted_image.png") }
#let fig8 = { image("../../figures/img/bg/hue_adjusted_image.png") }
#let fig9 = { image("../../figures/img/bg/noisy_image.png") }
#let fig10 = {image("../../figures/img/bg/image_with_square_erased.png") }

#std-block(breakable: false)[
  #figure(
    grid(
      columns: (1fr, 1fr, 1fr, 1fr, 1fr),
      fig1, fig2, fig3, fig4, fig5,
      [#subfigure("(a)")],[#subfigure("(b)")],[#subfigure("(c)")],[#subfigure("(d)")],[#subfigure("(e)")],
      fig6, fig7, fig8, fig9, fig10,
      [#subfigure("(f)")],[#subfigure("(g)")],[#subfigure("(h)")],[#subfigure("(i)")],[#subfigure("(j)")],
    ),
    caption: [Data augmentation techniques. (a) Original image. Applied augmentation are (b) cropping, (c) rotation, (d) flipping, (e) zooming, (f) blurring, (g) saturation adjustment, (h) hue adjustment, (i) noise addition, and (j) occlusion. Image source: viso.ai @cv_tasks.]
  ) <fig:data_augmentation>
]

The architectures, methods, and techniques present in modern deep learning and computer vision are vast and complex. However, these methods are only as effective as the data that fuels them. Selecting, generating, or collecting the right data for any given task is as crucial for the success of the model as the model itself. 






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
// Training: introduce occlusions (augmentation)

Datasets are the blood of any machine learning model. Entries flow through a model's structure, helping it learn to perform a specific task. In datasets, both the quantity and quality of the data are important. The more data a model has to learn from, the better it will perform. However, the quality and variety are also tantamount to the model's performance. If the data is not representative of the task, the model will not be able to learn to perform the task. If too little variety is present, it will not learn to generalize and only be able to perform the task on scenarios closely related to that which makes up its dataset. 

There exists many different datasets for very different purposes. As presented earlier, the ImageNet dataset was key to the success of AlexNet. This dataset consists of millions of images with many thousands of classes. For the task of image classification, this dataset was very sufficient in providing the model with enough data to learn and generalize from. However, the base ImageNet dataset is insufficient for related CV tasks, such as object detection and segmentation. For these tasks, the dataset needs to be annotated with bounding boxes and class labels for each object in the image. This is a very time-consuming task to do from the ground up, which is why it is often to search for datasets that are already annotated and fit the desired task.

Creating datasets of any significant size can be a rather difficult task, where some tasks are simpler than others. For image classification, each dataset entry only requires a single image and a class label. This is a fairly simple task to do, as the most difficult part comes from collecting the images, with labelling taking very little time per entry. For other tasks like object detection and segmentation, the task of labelling becomes much more difficult. This is because each entry requires a bounding box or segmentation mask for each object in the image. This is a very time-consuming thing to do, and the quality of the labels may be of worse quality than desired, meaning there also needs to be some form of quality control. This means that the creation will take even longer, as labels that are poorly made, might heavily affect the model's performance. 

To mitigate errors in the labelling process, some move to simply make the dataset with synthetic data. This is done by using a simulator of some kind, ranging from a simple game engine to a full-fledged simulator. This is done by creating a virtual world, where the objects in the world are simulated. This allows for the creation of a dataset with perfect labels, as the simulator knows exactly where each object is in the world. Furthermore, making synthetic data can help increase the diversity in a dataset for areas where it might be difficult to obtain through real-world means. The most common problem with generating synthetic data, especially imagery of real-world scenarios, is the fact that the generated data does not look like real-world data. This can cause problems for models trained on the data, as they might not be able to generalize to real-world data as textures and shadows are less likely to look realistic.

For AVs, the datasets are often more complex than just images. They often consist of multiple sensors, such as LiDAR, radar, and cameras. This balloons the size of the dataset, resulting in singular entries being multiple gigabytes in size. The Waymo Open Dataset #citation_needed is a prime example of this. It consists of 1950 entries, each consisting of 20 seconds of video at 10Hz. Each contains the data from a mid-range LiDAR, 4 short-range LiDARs, and 5 cameras. The entries contain labelled data for 4 object classes: vehicles, pedestrians, cyclists, and signs. @fig:datasets#subfigure("e") shows an example of some LiDAR readings overlaid on the corresponding camera image. Waymo is a very detailed dataset, with a lot of data to learn from. However, it is also a very large dataset, which makes it difficult to work with on consumer-level computers. Competitors do exist, such as the nuScenes #citation_needed dataset and the Argoverse #citation_needed dataset. 

Closely related are the lane-level datasets. These datasets are concerned with detecting and classifying lanes in images. This is a very important task for AVs, as it allows them to understand the road topology and navigate accordingly. The TuSimple dataset example shown in @fig:datasets#subfigure("d") illustrates this task well, providing annotated lane markings for training and evaluation. Systems like #acr("AIM") that manage the vehicle going through an intersection, datasets like the INTERACTION #citation_needed dataset are needed. This dataset consists of overhead views of intersections, with annotated vehicles moving through them. This dataset is very useful for training models to understand the interactions between vehicles at intersections. 

Finally, some satellite image-based datasets also exits for different purposes. The SpaceNet #citation_needed dataset is concerned with the labelling of buildings and roads etc. in satellite images. This dataset consists of high-resolution satellite images and different annotations for the images, such as individual houses, both through bounding boxes and segmentation masks. For a bit wider scale, the DeepGlobe #citation_needed dataset offers semantic masks of roads, building, and land cover.

#let fig1 = { image("../../figures/img/bg/waymo.png") }
#let fig2 = { image("../../figures/img/bg/deepglobe.png") }
#let fig3 = { image("../../figures/img/bg/lane.png") }
#let fig4 = { image("../../figures/img/bg/traj_pred.png") }
#let fig5 = { image("../../figures/img/bg/spacenet.png") }

#std-block(breakable: false)[
  #figure(
    grid(
      columns: (1fr),
      grid( columns: (1.11fr, 1fr), 
        fig5, fig4, 
        [#subfigure("(a)") SpaceNet #citation_needed], [#subfigure("(b)") INTERACTION #citation_needed]),
      grid( columns: (1fr, 0.73fr, 1.09fr), 
      fig2, fig3, fig1, 
      [#subfigure("(c)") DeepGlobe #citation_needed], [#subfigure("(d)") TuSimple #citation_needed], [#subfigure("(e)") Waymo #citation_needed])
    ),
    caption: [Examples from datasets for AVs. ]
  ) <fig:datasets>
]

Common for all datasets, is the fact that they require a dataloader in order to be used in the program training the model. A dataloader is a class that is responsible for loading the data from the dataset and preparing it for training. This means that the dataset needs to be structured in a consistent way, such that the dataloader can work as seamlessly with it as possible. A dataset typically consists of a number of entries, each with their own attributes, such as ground truth labels, images, LiDAR and radar data, and other sensor data. Maintaining a consistent structure across all entries is important, otherwise a dataloader might get needlessly complex with unnecessary fail-safes. 

Dataloaders often come with the ability to split a dataset into training and validation sets. This is done by randomly splitting the dataset into two parts, where one part is used for training and the other part is used for validation. This is commonly done by splitting it with a certain percentage going to either set. Alternatively, the dataset can be split beforehand, meaning it consists of predefined training and validation sets. This can be a good choice when the model could become prone to overfitting, which is likely to happen when the dataset entries are similar looking, such as with intersections. If techniques like cross-validation are used, the model is at some point trained on every entry, which for long training sessions will eventually lead to overfitting. This can, however, be mitigated with the use of data augmentation.

== Satellite Imagery <c2s2.3:si>
// Google Maps Static API, Azure Maps, Sentinel, OpenStreetMap
// Works even when cloudy (non-reliance on live sat images)
// Resolution
// HD maps

Datasets can consist of many different views. Many datasets for AVs consist of images taken from a car driving around, typically consisting of many different cameras and sensors. Others, like is the focus in this project, consist of satellite images. Satellite imagery provides a bird's eye view of the world. This is a very useful perspective for many tasks, such as road extraction and lane detection. Many different sources of satellite imagery exist, such as Google Maps, Azure Maps, and Sentinel, each offering their own capabilities through #acrpl("API"). The most important among these is the ability to get high-resolution static images of a specific location, preferably as close in dates as possible. This not only increases the likelihood of images being of a usable resolution, but also reflecting of the current state of locations.

The resolution of satellite images refers to the size of the pixels in the image. The less area a pixel covers, the more information and detail about the location is captured. Google Maps Static API offers a sub-meter spatial resolution, meaning each pixel represents about 30-50cm on the ground @sat_res. At this level of detail, things like road markings and lane boundaries can be detected. In contrast, if the resolution is at a meter-level spatial resolution, then these fine details are at best blurred and at worst completely lost since they represent potentially a fraction of the land making up said pixel. At this level, only the general structure may be inferred from the resulting satellite image. Thus, having a decently high-resolution image is very important for giving NNs a chance at understanding intersections. For example, it needs to be able to see road-marking that make up the lanes, especially important when certain lanes are for going certain ways, often marked by arrows; you would not want a vehicle go straight through an intersection when it has placed itself in a lane that is only for turning right.

Satellite images provide strong advantages to AVs. Firstly, they are not reliant on live images, meaning they can be used even when it is cloudy. This means that they can provide a consistent and reliable source of information about the environment, regardless of weather conditions. Secondly, they can help AVs understand the road topology of any intersection before arriving at it. This means that a vehicle can place itself in the appropriate lane beforehand, delivering a smoother experience to the driver and passengers. This is especially important for AVs, as they need to be able to navigate complex intersections without human intervention. Finally, satellite images can be used to create high-definition maps of the environment. These maps can be used to help AVs navigate and understand their surroundings, potentially even help them find paths that offer smoother rides or other desired traits of travelling to some destination.

In summary, this chapter has thoroughly established the theoretical foundation necessary for the work presented in this thesis. By exploring the taxonomy of autonomous vehicles and the technologies enabling their development, it has provided context for the broader application landscape. The in-depth overview of deep learning — from its historical roots to modern architectures — lays the groundwork for understanding the computational methods used in this project. Furthermore, the exploration of datasets and the unique role of satellite imagery highlights the importance of data quality and perspective in training effective models. With this background in place, the thesis now transitions into more closely related works before moving onto the more specific methodologies and experiments that form the core of this work.



//== Path Drawing <c2s2.4:path_drawing>
// Waypoints, trajectory smoothing, Bézier curves, splines

//== Pose Estimation <c2s2.5:pose_estimation>
// Potential for future work
// ensure correctness of vehicle following path