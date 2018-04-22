# RL-cmps590
Course project
---
author:
- 'Yuewei Yang, Mingxi Cheng'
title: Reinforcement Leanring in Image Captioning
---

Introduction
============

Image captioning is a challenging task in computer vision. It is
attracting increasing attention because describing the complicated
content of an image in a natural language is one of core human
intelligence and now a machine can be trained to do as good as a human.
In previous work, advanced algorithms have already overcome the
difficulties in extracting visual information from an image, deriving
textual content and sequence, and combining two pieces together to
compute a sequence of words describing an image. The most popular
algorithm consists a convolutional neural network to encode visual
information and a recurrent neural network to decode a sequence of words
from the encoded visual information [@show; @and; @tell].\
Reinforcement learning has been applied in many applications such as
gaming, robotics control, and finance. The core idea of how an agent
maximises total reward by interacting with its environment forms a
general framework for reward-based learning. It can be applied to any
learning problems. The structure of the model and the algorithm used in
this paper are designed based on [@RL; @image; @captioning]. The model
applied in this paper is simpler than proposed in
[@RL; @image; @captioning]. A difference in performance is expected and
will be explained in detail.\
An encoder-decoder based image captioning model [@merge; @model] is
implemented as a baseline. Based on [@merge; @model],
[@show; @and; @tell; @attention] and [@RL; @image; @captioning] a policy
network is computed and is used to update the policy using policy
gradient algorithm. A value network is further added to the model to
implement an actor-critic model so that both value and policy can be
optimized using corresponding gradients.\
The experiment is conducted using Flickr8k datasets and the performances
are compared in standard evaluation metrics: BLEU [@BLEU]. In this
paper, the aim is to study how to apply reinforcement learning in image
captioning and learn the basics of reinforcement learning using image
captioning as an application.

Related Work
============

In early works like [@bottom-up], it proposes a bottom-up model that
generates words from object recognition and attribute prediction, then
reconstructs a meaningful description using a language model. In more
recent studies, an encoder-decoder model is proposed using multiple
neural networks and is convinced to improve the accuracy of the
automated descriptions. The basic idea is to encode visual information
through a convoluted neural network and then together with a text
embedding model a sequence of texts are decoded through a recurrent
neural network. Most of the modern applications consist the
encoder-decoder learning model and more advanced algorithms involve
combining both models [@deep; @visual-semantic] or adding attention
mechanism to the model
[@semantic; @attention][@show; @and; @tell; @attention] so that the
performance can be improved further.\
In the past two years, more papers have discussed the validity of using
a reward based learning algorithm to caption an image.
[@policy; @RL; @1] [@policy; @RL; @2] have applied policy gradient
search to update the transition matrix/policy network and the results
show better performance. In most recent paper
[@RL; @image; @captioning], it proposed a value network based on current
total reward, a new visual-semantic embedding reward, and an inference
mechanism. By using actor-critic reinforcement learning, the model
outperforms most state-of-the-art approaches consistently in all scoring
metrics.

Reinforcement Learning Problem Formulation
==========================================

We formulate image captioning as a decision-making process. In
decision-making, there is an agent that interacts with the environment,
and executes a series of actions, so as to optimize a goal. In image
captioning, the goal is, given an image $I$, to generate a sentence
$S={w_1,w_2,...,w_T}$ which correctly describes the image content, where
$w_i$ is a word in sentence $S$ and $T$ is the length. Our model,
including the policy network $p_\pi$ and value network $v_\theta$, can
be viewed as the agent; the environment is the given image $I$ and the
words predicted so far ${w_1,...,w_t}$; and an action is to predict the
next word $w_{t+1}$. To measure the level achieved at each time step, a
reward representation is used. In this project, we decided to use two
different rewards: the loss at each time step as the cost and Euclidean
distance between text produced at each time step and the original text.
The effect of using these different reward representations will be
examined.

Models
======

In this section three different models will be presented: an
encoder-decoder model (baseline), a policy network model, and a
policy+value networks model. The ultimate goal of this project is to
achieve a simple policy+value networks using a simple reward
representation.

Encoder-Decoder Model
---------------------

In this model, image features are extracted through a convoluted neural
network (VGG16)[@VGG16], and text embeddings are extracted using a
recurrent neural network (LSTM)[@LSTM]. Figure 3.1 illustrates the
outline of the model. Since this model serves as a baseline, for more
details refer to [@merge; @model].

![Encoder-Decoder model
structure[]{data-label="fig:framework"}](model.png){width="75.00000%"}

The network on the top left branch is a text embedding network. Its
input is a word and the output is the corresponding embedding. The brach
on the top right is a visual information encoder. Its input is an image
and the output is a vector representation of the image. Then two
branches are joined into a decoder network, a forward feed layer. And
the final output is the next word. The definitions of RNN and LSTM used
in this model is well explained in [@RNN] and [@LSTM]. To train the
model, the cross-entropy loss is minimized:
$$-\log{p(S|I)} = -\sum^N_{t=0}{\log{p(S_1,S_2,...S_t|I)}}$$ where $S_t$
is the word embedding generated at each time step $t$, and $I$ is the
encoded visual information.

Policy Network Model
--------------------

In this model, the decoder part of the previous model is modified to
include an additional LSTM stage. With this additional stage, the
information of image and text is fed back to LSTM together at every time
step to update the hidden states in LSTM so the model could “infer” the
next most possible word. The output of the model is a probability matrix
of next possible words.

![Policy network
strucutre[]{data-label="fig:policy network"}](model_policy.png){width="100.00000%"}

![Illustration of policy network flow
[@RL; @image; @captioning][]{data-label="fig:policy illustration"}](policy_model.png){width="100.00000%"}

The visual information is fed into the initial input node
$x_0\in \mathbb{R}^n$ of RNN. As the hidden state $h_t\in\mathbb{R}^m$
of RNN evolves over time $t$, the policy at each time step to take an
action $a_t$ is provided. The generated word $w_t$ at $t$ will be fed
back into RNN in the next time step as the network input $x_{t+1}$,
which drives the RNN state transition from $h_t$ to $h_{t+1}$.
Specifically, the main working flow of policy network, $p_\pi$ is
governed by the following equations: $$\begin{aligned}
&x_0 = W\times{CNN(I)} \\
&h_t = RNN(h_{t-1,x_t}) \\
&x_t = \phi(w_{t-1},t>0) \\
&p_\pi(a_t|s_t) = \varphi(h_t)\end{aligned}$$ where $W$ is the weight of
the linear embedding model of visual information.$\phi$ and $\varphi$
denote the input and output models of RNN. $CNN$ and $RNN$ are the
visual information encoder netwrok and infer network as shown in Figure
3.

The model is first trained as a supervised learning, and the entropy
loss is to be minimised
$-\log{p(S|I)} = -\sum^N_{t=0}{\log{p(S_t|I,S_1,...,S_{t-1})}}$. Then
use reinforcement learning to maximise the total reward
$J = \mathbb{E}_{S_1...T~p_\pi}(r)$. Then optmised the model using
policy gradient search [@RL; @image; @captioning]:
$$\nabla{J_\pi}\approx{\sum_{t=1}^N{\nabla_{\pi}\log{p_\pi(a_t|s_t)}}}$$
So the probability matrix $p(a_t|s_t)$ is update after reinforcement
learning.

Policy and Value Networks Model
-------------------------------

The policy netwrok in this model is exactly the same as the one in
section 4.2. The value network is added in this model to implement an
actor-critic model. The structure of value network is shown below:

![Value network
structure[]{data-label="fig:value network"}](MLP.png){width="75.00000%"}

![Illustration of value network flow
[@RL; @image; @captioning][]{data-label="fig:value illustration"}](valuenetwork.png){width="100.00000%"}

The value network is an approximation of estimated total reward at time
step $t$: $v_\theta(s)\approx\mathbb{E}[r|s_t=s,a_{t,...T}\sim{p}]$,
where $s_t=\{I,w_1,...,w_t\}$ and $p$ is the policy taken. To train the
value network, first the mean square loss, $||v_\theta(s_i)-r||^2$, is
to be minimised. And then apply a gradient search in the value network.
So the reinforcement learning of this actor-critic model optimises the
model with two graidents: $$\begin{aligned}
\nabla{J_\pi}&\approx{\sum_{t=1}^N{\nabla_{\pi}\log{p_\pi(a_t|s_t)(r-v_\theta(s_t))}}} \\
\nabla{J_\theta}&=\nabla_{\theta}v_\theta(s_t)(r-v_\theta(s_t))\end{aligned}$$

Here the value network $v_\theta$ serves as a moving baseline.The
quantity $r-v_\theta(s_t)$ used to scale the gradient can be seen as an
estimate of the advantage of action $a_t$ in state $s_t$. This approach
can be viewed as an actor-critic architecture where the policy $p_\pi$
is the actor and $v_θ$ is the critic.

Experiments
===========

All three models are trained and tested on the Flickr8k dataset. There
are 8,097 images (6,000 training images, 1,000 validation images, and
1,000 test images) in the dataset. The reason for this small size
dataset is to control training time as COCO dataset has 123,287 images
and every single image increases the training time considerably. The
performance of each model is compared in terms of BLEU scores against
each other.

Results and Discussions
=======================

This section will demonstrate the performances of different models in
terms of BLEU scores. The effect of different reward representations on
the performance will be discussed too.

Models
------

  Methods                    Bleu-1   Blue-2   Bleu-3   Bleu-4
  -------------------------- -------- -------- -------- --------
  Encoder-Decoder            0.518    0.334    0.206    0.132
  Policy Network             0.544    0.362    0.231    0.151
  Policy and Value Network   0.551    0.378    0.251    0.169

Encoder-decoder model is a supervised learning model. The performance is
optimized by tuning the length of features and layer parameters shown in
Figure 1. Policy network model update transition probability matrix
$p_{\pi}(a_t|s_t)$ using policy gradient search. Policy and value
network model are implemented based on [@RL; @image; @captioning]. Using
reinforcement learning on policy network alone is very effective and it
is much better than the supervised learning model. This illustrates that
transition matrix has been improved by using policy gradient method and
the original matrix delivers poor result due to lack of training
samples. The policy and value network model perform slightly better than
policy network model. In [@RL; @image; @captioning], experiments
investigating the effect of value network alone are conducted. The
results show that the effect of the value network is not significant
compared with that of policy network. In the figure below shows some
images with descriptions generated.

![Examples of generated descriptions of
images[]{data-label="fig:examples"}](pic.png){width="100.00000%"}

The descriptions generated using supervised learning make error often.
Those generated using reinforcement learning are much better, though
policy network and policy and value network produce similar
descriptions. However, my best models do make errors (figure 6(d)). This
is due to the fact the best score can be achieved is about 0.551 in
BLEU-1. The scale in BLEU scores measures the accuracy of generated text
compared with original text. As a fact, dog picutures have good captions
through our model. But other picutres showed poor results. Futhermore,
as shown in the figure 6(a) above, there are some objects in the
original text that are not recognized by our model. And supervised
learning seems to miss more objects.

Reward Representations
----------------------

  Representations      Bleu-1   Blue-2   Bleu-3   Bleu-4
  -------------------- -------- -------- -------- --------
  Loss                 0.551    0.378    0.251    0.169
  Euclidean Distance   0.492    0.325    0.198    0.137

The model used in this experiment is policy and value network model.
Different reward representations do make a difference. Euclidean
distance between texts generated and original text does not perform as
well as loss cost. The explaination would be that our model cannot
recognise as many objects as in original text. Hence the distance
between them is going to be big. This could introduce some unstabability
into the system as $r-v_\theta(s_t)$ in equations (7) and (8) introduces
high variance. In [@RL; @image; @captioning] another reward
representation using visual-sematic embedding that maps two embeddings
into one. The effect of utilising combined embedding reward improves the
performance.

Future Work
===========

The best performance of our model is not achieving as good as other
state-of-art methods. One issue with experiments in this project is that
they train on Flickr8k and the limited samples cause inaccuracy in the
performance. Another defect of our model is that there is no inference
mechanism such as beam search used in most methods. In our models, we
only used the best choice only at each time step. In
[@RL; @image; @captioning], besides beam search, an lookahead mechanism
used as another inference mechanism. Policy network serves as a global
guidance and value network as a local guide. By applying different
wrights on these two guidances, the combined beam may have different
orders. This enables the model to include the good words that are with
low probability to be drawn by using the policy network alone. Another
area can improve the model is investigating other reward representations
and a better structure of the model.

Conclusion
==========

In this project, reinforcement learning is applied to image captions.
The reward-based learning is added to supervised learning model to
improve the transition matrix. In reinforcement learning models, a
policy network and a value network are trained first with supervised
learning and then update using a gradient method. The effect of policy
network is more significant than that of the value network. From this
project, the effect of reinforcement learning based model is compared
with supervised learning model. The effect of different reward
representation is also studied. How reinforcement learning can be
applied in addition to a supervised learning model is learned and
discussed.

Acknowledgment
==============

Complete codes for our model can be found on GitHub [^1]. In this
project, codes are implemented based on [@RL; @image; @captioning] and
[@show; @and; @tell]. Codes for reinforcement learning are modified
using [@xinping] and [@tsenghuangchen]. My consultant, Mingxi Cheng,
provides great assistance in constructing the model and writing codes.

[9]{} Vinyals, Oriol, et al. “Show and tell: A neural image caption
generator.” *Proceedings of the IEEE conference on computer vision and
pattern recognition*. 2015.

Ren, Zhou, et al. “Deep Reinforcement Learning-based Image Captioning
with Embedding Reward.” *arXiv preprint arXiv:1704.03899* (2017).

Tanti, Marc, Albert Gatt, and Kenneth P. Camilleri. “Where to put the
Image in an Image Caption Generator.” *arXiv preprint arXiv:1703.09137*
(2017).

Xu, Kelvin, et al. “Show, attend and tell: Neural image caption
generation with visual attention.” *International Conference on Machine
Learning*. 2015.

Papineni, Kishore, et al. “BLEU: a method for automatic evaluation of
machine translation.” *Proceedings of the 40th annual meeting on
association for computational linguistics*. Association for
Computational Linguistics, 2002.

Farhadi, Ali, et al. “Every picture tells a story: Generating sentences
from images.” *European conference on computer vision*. Springer,
Berlin, Heidelberg, 2010.

Karpathy, Andrej, and Li Fei-Fei. “Deep visual-semantic alignments for
generating image descriptions.” *Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition*. 2015.

You, Quanzeng, et al. “Image captioning with semantic attention.”
*Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition*. 2016.

Liu, Siqi, et al. “Optimization of image description metrics using
policy gradient methods.” *arXiv preprint arXiv:1612.00370* (2016).

Liu, Siqi, et al. “Improved Image Captioning via Policy Gradient
optimization of SPIDEr.” *arXiv preprint arXiv:1612.00370* (2016).

Simonyan, Karen, and Andrew Zisserman. “Very deep convolutional networks
for large-scale image recognition.”*arXiv preprint arXiv:1409.1556*
(2014).

Graves, Alex, and Jürgen Schmidhuber. “Framewise phoneme classification
with bidirectional LSTM and other neural network architectures.” *Neural
Networks* 18.5 (2005): 602-610.

Mao, Junhua, et al. “Deep captioning with multimodal recurrent neural
networks (m-rnn).” *arXiv preprint arXiv:1412.6632* (2014).

Cheng, Xinping, “Optimization of image description metrics using policy
gradient methods”,
<https://github.com/chenxinpeng/Optimization_of_image_description_metrics_using_policy_gradient_methods>

Chen, Tseng-Huang, “Show, Adapt and Tell: Adversarial Training of
Cross-domain Image Captioner”,
<https://github.com/tsenghungchen/show-adapt-and-tell>

[^1]: https://github.com/yueweiyang/RL-cmps590
