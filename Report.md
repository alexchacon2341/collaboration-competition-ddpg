[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135612-cbff24aa-7d12-11e8-9b6c-2b41e64b3bb0.gif "Trained Agent"
[image2]: https://lh3.googleusercontent.com/-QrAga9tv-Cc/XDzSj06OyHI/AAAAAAAAGE0/LEj_Vhkoj6whz364EEdYtWJyziDh41rvACL0BGAs/w530-d-h76-n-rw/Screen%2BShot%2B2019-01-14%2Bat%2B1.17.23%2BPM.png "DPG Algorithm"
[image3]: https://lh3.googleusercontent.com/-LKAjjGLELyw/XDzVZ56AIBI/AAAAAAAAGGE/vNo3E7Z1wmI9Q5XwInKWIdE_WeCn4pHrgCL0BGAs/w530-d-h350-n-rw/Screen%2BShot%2B2019-01-14%2Bat%2B1.29.19%2BPM.png "DDPG Algorithm"
[image4]: https://lh3.googleusercontent.com/-mBZhL8EN4Oc/XDzUSKwlkWI/AAAAAAAAGFY/13WHIZ9AomcdHgD49_ETahtlOvjvGVd_QCL0BGAs/w530-d-h85-n-rw/Screen%2BShot%2B2019-01-14%2Bat%2B1.25.55%2BPM.png "Exploration Policy"
[image5]: https://lh3.googleusercontent.com/-y8LZqmVuCW8/W4ToZiIV8bI/AAAAAAAAF7s/21hHC4Z9KKQZBwalr52NQyn9LLRCoiZPACL0BGAs/w530-d-h260-n-rw/Screen%2BShot%2B2018-08-28%2Bat%2B2.14.30%2BAM.png "Hyperparameters"
[image6]: https://lh3.googleusercontent.com/-GNL6JuAk98o/W4TsEVegb8I/AAAAAAAAF9A/fk9NXU8iXKwy4Ukxe0VjzxIeNF1qKa6UwCL0BGAs/w530-d-h359-n-rw/Screen%2BShot%2B2018-08-28%2Bat%2B2.30.05%2BAM.png "Plot"

# Report

### Methodology

The project uses methods involving deep neural networks developed in a [2016 paper](https://arxiv.org/pdf/1509.02971.pdf) to
creat an artificial agent that learns using a deep deterministic policy gradient (DDPG), which
uses end-to-end reinforcement learning to solve an environment created by Unity's ML-Agents. The architecture used in this case is PyTorch's nn Module, a deep recurrent
neural network (RNN) that is adept at defining computational graphs and taking gradients and is better for defining complex networks than raw autograd.

The agent interacts with its environment through a sequence of observations, 
actions, and rewards. Its goal is to select actions in order to
maximize cumulative future reward, as is standard in Q-Learning. However, since the environment contains a continuous action space (the amount of rotation applied to achieve a desired arm orientation), it is not possible to straightforwardly apply Q-learning, because in continuous spaces finding the greedy policy requires an optimization of at at every timestep. As such, here we used an actor-critic approach based on the DPG (deterministic policy gradient) algorithm. The maintains a parameterized actor function µ(s|θ
µ) which specifies the current
policy by deterministically mapping states to a specific action. The critic Q(s, a) is learned using
the Bellman equation as in Q-learning. The actor is updated by following the applying the chain rule
to the expected return from the start distribution J with respect to the actor parameters:

![DPG Algorithm][image2]

As with Q learning, introducing non-linear function approximators means that convergence is no
longer guaranteed. However, such approximators appear essential in order to learn and generalize
on large state spaces. Here, the effort is made to use DPG with neural network function approximators to learn in large
state and action spaces online. The resulting algorithm is DDPG:

![DDPG Algorithm][image3]

One challenge when using neural networks for reinforcement learning is that most optimization algorithms assume that the samples are independently and identically distributed. Obviously, when
the samples are generated from exploring sequentially in an environment this assumption no longer
holds. Additionally, to make efficient use of hardware optimizations, it is essential to learn in minibatches, rather than online.
As in DQN, we used a replay buffer to address these issues. The replay buffer is a finite sized cache
R. Transitions were sampled from the environment according to the exploration policy and the tuple
(st, at, rt, st+1) was stored in the replay buffer. When the replay buffer was full the oldest samples
were discarded. At each timestep the actor and critic are updated by sampling a minibatch uniformly
from the buffer. Because DDPG is an off-policy algorithm, the replay buffer can be large, allowing
the algorithm to benefit from learning across a set of uncorrelated transitions.
Directly implementing Q learning (equation 4) with neural networks proved to be unstable in many
environments. Since the network Q(s, a|θ
Q) being updated is also used in calculating the target
value (equation 5), the Q update is prone to divergence. Our solution is similar to the target network
used in (Mnih et al., 2013) but modified for actor-critic and using “soft” target updates, rather than
directly copying the weights. We create a copy of the actor and critic networks, Q0
(s, a|θ
Q0
) and
µ
0
(s|θ
µ
0
) respectively, that are used for calculating the target values. The weights of these target
networks are then updated by having them slowly track the learned networks: θ
0 ← τθ + (1 −
τ )θ
0 with τ  1. This means that the target values are constrained to change slowly, greatly
improving the stability of learning. This simple change moves the relatively unstable problem of
learning the action-value function closer to the case of supervised learning, a problem for which
robust solutions exist. We found that having both a target µ
0
and Q0 was required to have stable
targets yi
in order to consistently train the critic without divergence. This may slow learning, since
the target network delays the propagation of value estimations. However, in practice we found this
was greatly outweighed by the stability of learning.
When learning from low dimensional feature vector observations, the different components of the
observation may have different physical units (for example, positions versus velocities) and the
ranges may vary across environments. This can make it difficult for the network to learn effectively and may make it difficult to find hyper-parameters which generalise across environments with
different scales of state values.
One approach to this problem is to manually scale the features so they are in similar ranges across
environments and units. We address this issue by adapting a recent technique from deep learning
called batch normalization (Ioffe & Szegedy, 2015). This technique normalizes each dimension
across the samples in a minibatch to have unit mean and variance. In addition, it maintains a running average of the mean and variance to use for normalization during testing (in our case, during
exploration or evaluation). In deep networks, it is used to minimize covariance shift during training,
by ensuring that each layer receives whitened input. In the low-dimensional case, we used batch
normalization on the state input and all layers of the µ network and all layers of the Q network prior
to the action input (details of the networks are given in the supplementary material). With batch
normalization, we were able to learn effectively across many different tasks with differing types of
units, without needing to manually ensure the units were within a set range.
A major challenge of learning in continuous action spaces is exploration. An advantage of offpolicies algorithms such as DDPG is that we can treat the problem of exploration independently
from the learning algorithm. We constructed an exploration policy µ
0 by adding noise sampled from
a noise process N to our actor policy

![Exploration Policy][image4]

N can be chosen to chosen to suit the environment. As detailed in the supplementary materials we
used an Ornstein-Uhlenbeck process (Uhlenbeck & Ornstein, 1930) to generate temporally correlated exploration for exploration efficiency in physical control problems with inertia (similar use of
autocorrelated noise was introduced in (Wawrzynski, 2015)).

### Hyperparameters

To best compare across environments, the hyperparemeters used to generate the experiences in "nav_weights.pth" were similar to those used in the paper on which the algorithm was based. The algorithm from this research was able to a achieve a level of performance comparable to that of a professional human games tester across a set of 49 Atari games using only one set of hyperparameters, and these hyperparameters were imitated to attempt similar results while using an RNN as opposed to a CNN (Convolutional Neural Network). Precise values and descriptions for each hyperparameter follow:

![Hyperparameters][image5]

Using these settings, the environment was solved in 497 episodes with an average consecutive reward of +13.01. The following plot shows the agent's progress throughout the training session:

![Plot][image6]

### Suggestions

While the agent was able to converge on a policy that solved the environment in a relatively short period of time, Additional changes may still yield improvements. To increase the likelihood that the agent continues exploring different actions until the optimal policy has been found, it may be beneficial to implement gamma with a curriculum-type structure, ensuring its decay pauses at certain thresholds until a pre-specified average reward is reached. Increasing the number of hidden layers may also yield better results. Finally, extensions of the DQN algorithm, including Double DQN, Dueling DQN, or Rainbow may converge on a more optimal policy in a shorter period.
