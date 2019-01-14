[//]: # (Image References)

[image1]: https://lh3.googleusercontent.com/-QrAga9tv-Cc/XDzSj06OyHI/AAAAAAAAGE0/LEj_Vhkoj6whz364EEdYtWJyziDh41rvACL0BGAs/w530-d-h76-n-rw/Screen%2BShot%2B2019-01-14%2Bat%2B1.17.23%2BPM.png "DPG Algorithm"
[image2]: https://lh3.googleusercontent.com/-LKAjjGLELyw/XDzVZ56AIBI/AAAAAAAAGGE/vNo3E7Z1wmI9Q5XwInKWIdE_WeCn4pHrgCL0BGAs/w530-d-h350-n-rw/Screen%2BShot%2B2019-01-14%2Bat%2B1.29.19%2BPM.png "DDPG Algorithm"
[image3]: https://lh3.googleusercontent.com/-y8LZqmVuCW8/W4ToZiIV8bI/AAAAAAAAF7s/21hHC4Z9KKQZBwalr52NQyn9LLRCoiZPACL0BGAs/w530-d-h260-n-rw/Screen%2BShot%2B2018-08-28%2Bat%2B2.14.30%2BAM.png "Hyperparameters"
[image4]: https://lh3.googleusercontent.com/-GNL6JuAk98o/W4TsEVegb8I/AAAAAAAAF9A/fk9NXU8iXKwy4Ukxe0VjzxIeNF1qKa6UwCL0BGAs/w530-d-h359-n-rw/Screen%2BShot%2B2018-08-28%2Bat%2B2.30.05%2BAM.png "Plot"

# Report

### Methodology

The project uses methods involving deep neural networks developed in a [2016 paper](https://arxiv.org/pdf/1509.02971.pdf) to
create an artificial agent that learns using a deep deterministic policy gradient (DDPG), which
uses end-to-end reinforcement learning to solve an environment created by Unity's ML-Agents. The architecture used in this case is PyTorch's nn Module, a deep recurrent
neural network (RNN) that is adept at defining computational graphs and taking gradients and is better for defining complex networks than raw autograd.

The agent interacts with its environment through a sequence of observations, 
actions, and rewards. Its goal is to select actions in order to
maximize cumulative future reward, as is standard in Q-Learning. However, since the environment contains a continuous action space (the amount of rotation applied to achieve a desired arm orientation), it is not possible to straightforwardly apply Q-learning, because in continuous spaces finding the greedy policy requires an optimization of at at every timestep. As such, an actor-critic approach based on the DPG (deterministic policy gradient) algorithm is used. This algorithm maintains a parameterized actor function µ(s|θ
µ) which specifies the current
policy by deterministically mapping states to a specific action. The critic Q(s, a) is learned using
the Bellman equation as in Q-learning. The actor is updated by following the applying the chain rule
to the expected return from the start distribution J with respect to the actor parameters:

![DPG Algorithm][image1]

As with Q learning, introducing non-linear function approximators means that convergence is no
longer guaranteed. However, such approximators appear essential in order to learn and generalize
on large state spaces. Here, the effort is made to use DPG with neural network function approximators to learn in large
state and action spaces online. The resulting algorithm is DDPG:

![DDPG Algorithm][image2]

One challenge when using neural networks for reinforcement learning is that most optimization algorithms assume that the samples are independently and identically distributed. Obviously, when
the samples are generated from exploring sequentially in an environment this assumption no longer
holds. Additionally, to make efficient use of hardware optimizations, it is essential to learn in minibatches, rather than online.

As in DQN, a replay buffer is used to address these issues. The replay buffer is a finite sized cache
R. Transitions were sampled from the environment according to the exploration policy and the tuple
(st, at, rt, st+1) was stored in the replay buffer. When the replay buffer was full, the oldest samples
were discarded. At each timestep, the actor and critic are updated by sampling a minibatch uniformly
from the buffer. Because DDPG is an off-policy algorithm, the replay buffer can be large, allowing
the algorithm to benefit from learning across a set of uncorrelated transitions.

Directly implementing Q learning with neural networks can be unstable in many
environments. Since the network Q(s, a|θ
Q) being updated is also used in calculating the target
value, the Q update is prone to divergence. This issue was addressed with a to the target network
 modified for actor-critic and using “soft” target updates, rather than
directly copying the weights. A copy of the actor and critic networks, Q0
(s, a|θ
Q0
) and
µ
0
(s|θ
µ
0
) respectively, were created and used for calculating the target values. The weights of these target
networks were then updated by having them slowly track the learned networks: θ
0 ← τθ + (1 −
τ )θ
0 with τ << 1. This means that the target values are constrained to change slowly, greatly
improving the stability of learning. This simple change moves the relatively unstable problem of
learning the action-value function closer to the case of supervised learning, a problem for which
robust solutions exist. The resulting learning was slow, since
the target network delayed the propagation of value estimations. In practice, however, its downsides were greatly outweighed by the stability of learning.

### Hyperparameters

To best compare across environments, the hyperparemeters used to generate the experiences in "nav_weights.pth" were similar to those used in the paper on which the algorithm was based. The algorithm from this research was able to a achieve a level of performance comparable to that of a professional human games tester across a set of 49 Atari games using only one set of hyperparameters, and these hyperparameters were imitated to attempt similar results while using an RNN as opposed to a CNN (Convolutional Neural Network). Precise values and descriptions for each hyperparameter follow:

![Hyperparameters][image3]

Using these settings, the environment was solved in 497 episodes with an average consecutive reward of +13.01. The following plot shows the agent's progress throughout the training session:

![Plot][image4]

### Suggestions

While the agent was able to converge on a policy that solved the environment, the learning process was quite slow, taking nearly five hours to complete. An extension of DDPG may still improve the learning rate. Although the environment was solved for a single agent, modifications that allow the DDPG model to train with multiple agents may speed learning, as the agents would be able to gather experiences concurrently and share their outcomes.
