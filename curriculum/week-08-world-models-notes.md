# World Models: Full Study Notes

# Part 1 — Why world models at all

### 1.1 The batter

Batters do not react to the ball — they **predict** it from the pitcher's kinematics and the first few tens of milliseconds of flight, and commit to a swing before the visual evidence arrives.

The pedagogical point: a forward model is not a luxury feature bolted onto perception. It is what makes control possible at all when sensing is slower than the world. The same argument transfers directly to a robot with a 30 Hz camera, 100 ms inference latency, and contact events that resolve in 10 ms.

### 1.2 Policies versus world models (slide 11)

| | Policy / VLA | World model |
| --- | --- | --- |
| Question | Given $s_t$ and goal $g$, what should I do? | Given $s_t$ and action $a_t$, what happens next? |
| Object | $\pi(a_t \mid s_t, g)$ | $p(s_{t+1} \mid s_t, a_t)$ |
| Output | action to execute | next state |
| Supervision | needs action labels (teleop) | needs only transitions — and for video models, no actions at all |
| Counterfactuals | none — one answer per state | native — evaluate *any* candidate action |

The lecture's framing: **policies/VLAs are blind to physical causality and temporal dynamics.** A VLM fine-tuned on static image–text pairs has never been given a gradient that says "if the gripper closes here, the object rotates there." It can learn the *correlation* between an instruction and a demonstrated action, which is why VLAs need enormous teleoperation datasets — the dynamics knowledge has to be re-derived from action labels, the most expensive supervision in robotics.

A world model factors the problem: dynamics can be learned from cheap data (video, play data, autonomous interaction, failures), and only the final action mapping needs expensive data.

### 1.3 A world model is a data-driven simulator (slide 13)

| Traditional simulator | World model |
| --- | --- |
| hand-engineered physics engine: rigid bodies, contact models, friction, collision geometry | real interaction data: robot trajectories, human video, internet-scale video |
| $s_{t+1} = f(s_t, a_t)$ — deterministic, hand-coded | $p(s_{t+1} \mid s_t, a_t)$ — learned from data |
| accurate only for modelled phenomena | generalises to anything present in the data |
| ✅ fast, repeatable, resettable, gives you ground-truth state | ✅ models anything capturable on video (cloth, granular media, liquids, deformables) |
| ❌ requires hand-engineered physics; sim-to-real gap | ❌ no ground-truth state; hallucinates off-distribution; not resettable in the same clean way |

Note the *deterministic vs. probabilistic* distinction in the notation, and take it seriously. The world is stochastic from the agent's point of view for two separable reasons:

- **Aleatoric** — genuinely unpredictable (which way a tipping bottle falls, what a human does next).
- **Epistemic** — predictable in principle, unknown to this model (mass of an unseen object, friction coefficient).

A deterministic model collapses both into a conditional mean. Under a multimodal future, the conditional mean is a state that never occurs — the physical analogue of blur. Which brings us to the first family's central failure.

### 1.4 What prediction buys you (slide 12)

$$
\underbrace{N \text{ imagined rollouts}}_{\text{free, parallel, in the model}} \;\longrightarrow\; \underbrace{1 \text{ real trial}}_{\text{expensive, irreversible}}
$$

The economics of robotics are dominated by the cost of real trials: wall-clock time, hardware wear, human supervision, and the fact that a failed trial can break the robot or the object. A world model converts a search problem in the real world into a search problem in a GPU. That is the entire value proposition, and it is why the quality metric that matters is *"does planning through this model improve real decisions"* — not reconstruction PSNR.

### 1.5 The strict definition (slide 14)

$$
p(s_{t+1} \mid s_t, a_t, h_t)
$$

> **Strictly, a text-to-video model $p(\text{video} \mid \text{text})$ is not a world model.**

This is worth 10 minutes of discussion because it is where the field's terminology is loosest. The criterion is **action-conditioning and causal controllability**: a world model must let you *intervene* — fix $s_t$, vary $a_t$, and get different, correct futures. A text-to-video model samples a plausible video from a prompt. It has no interface through which you can ask "what if the gripper had moved 3 cm left instead."

In Pearl's language: a world model must support $p(s_{t+1} \mid s_t, \mathrm{do}(a_t))$, not merely $p(s_{t+1} \mid s_t, a_t)$ as an observational conditional. Since the training data comes from some behaviour policy $\beta(a\mid s)$, and we condition on the action explicitly, the two coincide *provided* actions are recorded and there are no unobserved confounders driving both action and outcome. Where the behaviour policy conditions on something the model cannot see (an operator looking at a second camera), the learned "world model" will absorb operator intent into its dynamics and will mispredict under a new policy. This is a real and under-discussed failure mode of models trained on teleop data.

**Discussion prompt.** Where do the families sit on the "is it really a world model" spectrum? DreamZero jointly predicts actions and video — is the video branch a world model or a fancy prior? V-JEPA 2-AC is action-conditioned but predicts no pixels — is it more or less of a world model than Sora?

---

## Part 2 — Family I: pixel action-conditioned world models

**State = the image itself.** Predict the next frame given the current frame and action. Conceptually the most direct thing you could do, and historically first.

### 2.1 Finn, Goodfellow & Levine (2016): predict motion, not pixels (slide 15)

**Setup.** A conv-LSTM stack (7 LSTM layers between strided convolutions, 64×64×3 input) consumes $(o_t, a_t)$ plus the robot's internal state and outputs a transformation of the current frame.

**The key idea.** Do not regress pixel *values*. Regress a **motion field** and apply it to the frame you already have:

$$
\hat o_{t+1} \;=\; \hat F_{t+1 \leftarrow t} \diamond o_t
$$

where $\diamond$ denotes applying the predicted warp. Concretely for the DNA (Dynamic Neural Advection) variant, the network outputs, for each output pixel $(x,y)$, a normalised distribution $\hat m_{xy}(k,l)$ over a $\kappa \times \kappa$ neighbourhood of source pixels:

$$
\hat o_{t+1}(x,y) \;=\; \sum_{k \in (-\kappa,\kappa)} \sum_{l \in (-\kappa,\kappa)} \hat m_{xy}(k,l)\; o_t(x-k,\; y-l)
$$

The CDNA (Convolutional DNA) variant is the parameter-efficient version: instead of a per-pixel kernel, predict $M=10$ global $5\times5$ convolution kernels $\hat m^{(j)}$ applied to the whole image, plus per-pixel **compositing masks** $\Xi^{(j)}$ (softmax over $j$) that decide which transformation applies where:

$$
\hat o_{t+1} \;=\; \sum_{j=1}^{M} \Xi^{(j)} \odot \big( \hat m^{(j)} * o_t \big)
$$

**Why this is the right inductive bias.** Each kernel is naturally interpreted as "one rigidly moving object." The mask segments the scene into things that move together. The model never has to learn to *paint* a banana — it only has to learn *where the banana goes*. Consequences:

- **Appearance-invariance for free.** Because the model transports whatever pixels are there, it generalises to objects it has never seen. This is why the 2016 result was striking: the motion objective is object-agnostic.
- **Occlusion and disocclusion are the hard cases.** A pure warp cannot invent newly revealed background. The architecture includes a static background image in the compositing stack precisely to patch this.

**Training.** Self-supervised on 50k+ autonomously collected robot pushing trajectories, $\ell_2$ loss on the predicted frame. No human labels, no reward.

### 2.2 Visual Foresight / Visual MPC (Ebert, Finn et al. 2018) (slides 16–17)

Take that predictor and plan with it. The full loop:

```text
Training
  robot autonomous interaction
    → large-scale dataset of (o_t, a_t, o_{t+1})
    → pixel prediction model (conv-LSTM, ℓ2 loss, pixel warp)

Test time — Visual MPC
  o_t  (current obs)
    → sample N action sequences {a_{t:t+H}}
    → roll out pixel predictions ô_{t:t+H}
    → score with a cost function
    → keep best a*, execute FIRST action only
    → observe real o_{t+1}, replan
```

**The three cost functions (choose one):**

1. **Pixel distance to a designated goal.** The user clicks a pixel on the object and a target location. The model predicts, for each future frame, a *distribution over where the designated pixel went* (this is why the flow formulation matters — the same warp that moves pixels moves the designated-pixel distribution). Cost:

   $$
   C \;=\; \sum_{t'=t+1}^{t+H} \mathbb{E}_{(x,y)\sim \hat d_{t'}} \big[\, \| (x,y) - g \|_2 \,\big]
   $$

   where $\hat d_{t'}$ is the predicted distribution of the designated pixel at time $t'$ and $g$ the goal pixel.

2. **Goal image via registration.** Given a goal image $o_g$, register it to the current frame and derive designated pixels automatically.

3. **Learned classifier.** A few-shot success classifier scores predicted final frames.

**The optimiser: Cross-Entropy Method.** Gradient-free, trivially parallel, robust to the non-smooth cost surface of a pixel predictor. See [§7.2](#72-random-shooting-cem-and-mppi) for the math.

**Verdict on the family (slide 17):**

| ✅ | ❌ |
| --- | --- |
| Self-supervised on robot video — no reward labels | Blurry predictions under $\ell_2$ loss |
| Generalises to unseen objects thanks to the motion objective | Planning in pixel space is computationally expensive |
| Interpretable — you can *watch* the plan | Prediction error accumulates ($\mathcal{O}(\varepsilon H^2)$, §0.3) |

### 2.3 Why $\ell_2$ gives you blur — and why it matters

This deserves an explicit derivation, because "L2 makes it blurry" is usually stated as folklore.

Minimising expected squared error over a conditional distribution yields the conditional mean:

$$
\arg\min_{\hat y} \; \mathbb{E}_{y \sim p(\cdot \mid x)} \big[\, \|\hat y - y\|_2^2 \,\big] \;=\; \mathbb{E}[\, y \mid x \,]
$$

If the future is multimodal — the pushed bottle might tip left or right — the conditional mean is the *pixelwise average of both outcomes*: a semi-transparent bottle in two places. That image is not a possible future. It is not merely aesthetically bad; it is **dynamically wrong**, and a planner that scores it will get nonsense: the average of "success" and "failure" can score better than either.

Formally, the model has collapsed a distribution to a point, and $\mathbb{E}[f(y)] \neq f(\mathbb{E}[y])$ for the nonlinear costs planners use.

**The three escape routes**, which organise the rest of the lecture:

1. **Model the distribution properly in pixel space** → diffusion / flow matching (DIAMOND, §2.4; DreamZero, §4.5).
2. **Move to a latent space where the distribution is simple** → VAE + mixture density or categorical latents (Family II, Part 3).
3. **Stop predicting the observation at all** → JEPA (Family V, Part 5).

### 2.4 🔶 DIAMOND: pixel world models, done properly

*Alonso, Jelley, Micheli, Kanervisto, Storkey, Pearce, Fleuret — "Diffusion for World Modeling: Visual Details Matter in Atari" (NeurIPS 2024 Spotlight), [arXiv:2405.12399](https://arxiv.org/abs/2405.12399).*

DIAMOND is not on the slides but belongs in Part 2, because it is the direct answer to §2.3 and it reopens a question the field had considered closed. Its thesis: **compression into a compact discrete latent throws away visual details that matter for control**, and if you fix the blur problem, pixel-space world models are competitive again.

**The formulation.** Instead of a point prediction, learn a conditional diffusion model over the next frame, following the EDM parameterisation (Karras et al. 2022). Let $x^0 = o_{t+1}$ be the clean next frame and let the conditioning be the recent history

$$
c \;=\; \big( o_{t-L:t},\; a_{t-L:t} \big)
$$

Perturb with Gaussian noise of scale $\sigma$: $\;x^\sigma = x^0 + \sigma\,n,\; n \sim \mathcal{N}(0,I)$.

**Preconditioned denoiser.** Rather than have the network predict $x^0$ or $n$ directly (which is badly conditioned across noise scales), EDM wraps a raw network $F_\theta$:

$$
D_\theta(x^\sigma; \sigma, c) \;=\; c_{\text{skip}}(\sigma)\, x^\sigma \;+\; c_{\text{out}}(\sigma)\, F_\theta\big( c_{\text{in}}(\sigma)\, x^\sigma;\; c_{\text{noise}}(\sigma),\; c \big)
$$

with, for data of scale $\sigma_{\text{data}}$,

$$
c_{\text{skip}} = \frac{\sigma_{\text{data}}^2}{\sigma^2 + \sigma_{\text{data}}^2}, \qquad
c_{\text{out}} = \frac{\sigma \cdot \sigma_{\text{data}}}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}, \qquad
c_{\text{in}} = \frac{1}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}
$$

The intuition for the skip connection: at **low** $\sigma$, $c_{\text{skip}} \to 1$ — the answer is nearly the input, so the network only predicts a small correction. At **high** $\sigma$, $c_{\text{skip}} \to 0$ — the input is pure noise, so the network must predict the frame from the conditioning alone. Without this, a single network has to span both regimes at wildly different output scales.

**Training objective.**

$$
\mathcal{L}(\theta) \;=\; \mathbb{E}_{x^0,\, c,\, \sigma \sim p(\sigma),\, n} \Big[\; \big\| D_\theta(x^0 + \sigma n;\, \sigma, c) - x^0 \big\|_2^2 \;\Big]
$$

with $\ln \sigma \sim \mathcal{N}(P_{\text{mean}}, P_{\text{std}}^2)$. Note this **is** an $\ell_2$ loss — the multimodality is not resolved by changing the loss but by making the target *conditional on the noise sample*, so the learned object is the full score function rather than a mean:

$$
\nabla_{x} \log p(x; \sigma) \;=\; \frac{D_\theta(x; \sigma) - x}{\sigma^2}
$$

**Sampling** integrates the probability-flow ODE backwards in noise level:

$$
\frac{dx}{d\sigma} \;=\; \frac{x - D_\theta(x; \sigma, c)}{\sigma}
$$

DIAMOND's practical finding: with Euler steps and this parameterisation, **a handful of denoising steps (as few as 1–3) suffice** for world modelling — because the conditioning is so strong that the conditional distribution is nearly unimodal most of the time. This is what makes a diffusion world model fast enough to train an RL agent inside.

**Agent training.** An actor-critic is trained purely on imagined rollouts from the diffusion model, exactly the Dreamer recipe but with a pixel-space, diffusion-based dynamics model.

**Headline results.** Mean human-normalised score of **1.46 on Atari 100k** — best among agents trained entirely inside a world model at publication. Qualitatively, DIAMOND preserves small critical sprites (a distant enemy, a bullet) that discrete-token models drop; the paper's title claim is that these details are precisely what the RL agent needs.

**Session question.** DIAMOND says "visual details matter." DINO-WM (§5.4) and JEPA say "pixels are an unnecessary cost." Both report strong results. Are they in conflict, or are they answering different questions — Atari (dense visual reward signals, small images, no embodiment transfer) versus robot manipulation (goals specifiable in feature space, transfer across scenes)?

### 2.5 🔶⚠️ MIRA: interactive pixel world models at scale

*"MIRA: Multiplayer Interactive World Models with Representation Autoencoders" — General Intuition, Kyutai, Epic Games. Playable at [mira-wm.com](https://mira-wm.com/); code at [github.com/mira-wm/mira](https://github.com/mira-wm/mira); technical report [arXiv:2607.05352](https://arxiv.org/abs/2607.05352).*

⚠️ *Details below are from the project page and repository, not a full read of the technical report.*

MIRA is a **5B-parameter latent diffusion transformer** that generates Rocket League frame-by-frame, conditioned on the actions of all four players, running a full 2v2 match inside the model at **20 FPS on a single GPU**. Training data: roughly 10,000 hours of bot-generated 2v2 matches. Action conditioning is a per-frame multi-hot keyboard-state vector for each of the four players.

Two things make it relevant to this lecture:

**1. It is a genuine multi-agent action-conditioned world model.** The formula becomes

$$
p\big(o_{t+1} \mid o_{\le t},\, a^{(1)}_t, a^{(2)}_t, a^{(3)}_t, a^{(4)}_t\big)
$$

Multi-agent conditioning is a real structural difference, not a cosmetic one. With one agent, the model can partially cheat by learning $p(o_{t+1}\mid o_{\le t})$ and treating the action as a weak hint, because the behaviour policy makes actions predictable from observations. With four independently controlled agents, action-conditioning must actually work or the model desynchronises immediately. This makes MIRA a useful stress test for *controllability*, which is exactly what is hard to measure in single-agent video WMs.

**2. The "representation autoencoder" (RAE).** Instead of the standard VAE latent space, MIRA's codec (~600M params) is built on a **frozen DINOv3-L/16 encoder** with a trained decoder that inverts those features back to pixels. The diffusion model then generates in *semantic feature space* rather than a compression-optimal space.

This is the conceptual bridge to Part 5. A VAE latent is optimised for reconstruction — it allocates capacity by pixel variance. A DINO feature space is optimised for semantic discriminability — it allocates capacity by "what makes these scenes different." If you are going to generate anyway, generating in a representation space is a hedge that gets you some of JEPA's benefit while keeping a decoder for interpretability and for human play.

**Related, and worth flagging for the paper-presentation slot:** [MiraBench](https://arxiv.org/abs/2605.29360) — "Evaluating Action-Conditioned Reliability in Robotic World Models" — is directly about the measurement problem in §8.

---

## Part 3 — Family II: latent action-conditioned world models

> **The pivot question (slide 18):** *What if we compressed observations into a latent space and trained a policy entirely inside that learned world?*

**The general structure (slide 19):**

$$
\begin{aligned}
\text{encoder:}\quad & z_t = \mathrm{enc}_\phi(o_t), \quad z_t \in \mathbb{R}^d \\
\text{dynamics:}\quad & h_{t+1} = f(h_t, z_t, a_t) \\
& p(z_{t+1} \mid z_t, a_t, h_t) \\
\text{reward:}\quad & p(\hat r_t \mid h_t, z_t) \\
\text{policy:}\quad & a_t = \pi(z_t, h_t)
\end{aligned}
$$

**Training the policy in imagination:**

1. Fix encoder + dynamics model.
2. Roll out imagined $z_1, \ldots, z_H$ (no environment).
3. Policy acts on $(z_t, h_t)$.
4. Optimise the policy on imagined rewards $\hat r_t$.

⇒ **the policy improves without any real environment interaction.**

Why latents rather than pixels: (a) rollouts become cheap — a 1024-dim vector instead of a 64×64×3 image, so you can imagine thousands of trajectories in a batch; (b) the distribution over next-states is far easier to model in a well-chosen latent space than in pixel space; (c) you can backpropagate through the whole rollout, which pixels make numerically miserable.

### 3.1 The original: Ha & Schmidhuber (2018), V–M–C (slides 20–25)

Three components, each trained separately and in this order:

```text
o_t --[ V: VAE ]--> z_t --[ M: MDN-RNN ]--> h_t --[ C: linear ]--> a_t
                             ^                                       |
                             +---------------------------------------+
```

#### 3.1.1 V — the Vision model (VAE)

A convolutional VAE compresses each frame independently to $z_t \in \mathbb{R}^{32}$ (car racing). Trained with the standard ELBO:

$$
\mathcal{L}_{\text{VAE}} \;=\; \underbrace{\mathbb{E}_{q_\phi(z\mid o)}\big[\log p_\theta(o \mid z)\big]}_{\text{reconstruction}} \;-\; \underbrace{D_{\mathrm{KL}}\big(q_\phi(z \mid o) \,\|\, \mathcal{N}(0,I)\big)}_{\text{regulariser}}
$$

with $q_\phi(z\mid o) = \mathcal{N}(\mu_\phi(o), \operatorname{diag}\sigma^2_\phi(o))$, sampled by reparameterisation $z = \mu_\phi + \sigma_\phi \odot \epsilon$, $\epsilon \sim \mathcal{N}(0,I)$ (see [§A.1](#a1-the-elbo-and-the-reparameterisation-trick)).

Crucially: **V is trained on random-policy rollouts, per frame, with no notion of time, action, or reward.** Remember this — it is the flaw that RSSM fixes.

#### 3.1.2 M — the Memory model (MDN-RNN)

An LSTM predicts the *distribution* of the next latent given the current latent, action, and hidden state:

$$
p(z_{t+1} \mid a_t, z_t, h_t)
$$

The output head is a **Mixture Density Network**: a mixture of $K$ diagonal Gaussians,

$$
p(z_{t+1} \mid h_t) \;=\; \sum_{k=1}^{K} \pi_k(h_t)\; \mathcal{N}\big(z_{t+1};\; \mu_k(h_t),\; \operatorname{diag}\sigma_k^2(h_t)\big), \qquad \sum_k \pi_k = 1
$$

trained by maximum likelihood, i.e. minimising

$$
-\log \sum_{k} \pi_k(h_t)\, \mathcal{N}\big(z_{t+1}; \mu_k(h_t), \operatorname{diag}\sigma_k^2(h_t)\big)
$$

**Why a mixture and not a single Gaussian?** This is the §2.3 argument again, one level up. The future is multimodal — the car may or may not clip the corner — and a unimodal Gaussian would return the mean of the modes. The MDN gives the model an explicit way to say "one of these $K$ things."

**Temperature $\tau$.** Ha & Schmidhuber sample with a temperature that scales the mixture: $\sigma_k \to \tau\,\sigma_k$ and $\pi \to \operatorname{softmax}(\log \pi / \tau)$. Raising $\tau$ makes the dream *more uncertain*. The paper's most interesting empirical result is that a **higher-temperature dream produces controllers that transfer better to the real environment** — because a policy trained in an over-confident dream learns to exploit the model's certainties, while a noisier dream forces robustness. This is the earliest clean statement of the model-exploitation problem and an uncertainty-based cure.

#### 3.1.3 C — the Controller

Deliberately tiny: a single fully-connected layer

$$
a_t \;=\; W_c\, [\,z_t;\, h_t\,] + b_c
$$

with only a few hundred to a few thousand parameters. Trained by **CMA-ES**, a black-box evolutionary optimiser:

1. Sample $N$ controllers $W_c^{(i)} \sim \mathcal{N}(\mu, \Sigma)$.
2. For each candidate, run a **dream rollout** and collect imagined cumulative reward.
3. Select the top-$k$ rollouts.
4. Update $\mu$ and $\Sigma$ toward the top rollouts.

**Why evolution and not gradients?** Two reasons, one good and one historical. The good one: the controller is small enough that a population-based search over ~1k parameters is cheap and needs no differentiability. The historical one: nobody had yet worked out how to backprop cleanly through a stochastic latent rollout — which is exactly Dreamer V1's contribution (§3.4).

**Design principle worth stating out loud:** put almost all parameters in the world model (V and M, millions) and almost none in the policy (C, thousands). The credit-assignment problem is hard; the representation problem is where the data is. This principle recurs in every model in this lecture, and it is the opposite of the VLA design point.

#### 3.1.4 Results and the ablation that matters (slides 24–25)

- Car Racing solved from pixels; the first agent trained purely in imagination.
- **Remove the memory model $M$ and performance collapses.** $z_t$ alone is a per-frame code with no velocity, no momentum, no history. $h_t$ is what makes the state approximately Markov. This is §0.2 made empirical.

### 3.2 Why imagination without a prior drifts (slide 26)

The problem that motivates the entire RSSM line:

**During training**, the loop is
$$o_t \xrightarrow{\ \text{VAE}\ } z_t \xrightarrow{\ \text{MDN-RNN}\ } h_t = f(h_{t-1}, z_t, a_t) \xrightarrow{\ \text{MDN}\ } p(z_{t+1}\mid h_t)$$
— a real observation is always available, so $z_t$ is always a *real* VAE code.

**In imagination**, there is no $o_t$. The loop must be
$$\hat z_t \xrightarrow{\ \text{MDN-RNN}\ } h_t = f(h_{t-1}, \hat z_t, a_t) \xrightarrow{\ \text{MDN}\ } p(\hat z_{t+1}\mid h_t)$$
with the *hallucinated* $\hat z$ fed back in.

Three compounding problems:

1. **The VAE is trained separately from the RNN.** It encodes $o_t$ — a description of an observation — not the model's own belief. Nothing ties the VAE's latent geometry to what the dynamics model finds predictable.
2. **There is no prior $p(z_t \mid h_t)$.** The MDN gives you $p(z_{t+1}\mid h_t)$ one step ahead, but there is no principled way to form a belief about the *current* state from memory alone. The model has no internal compass.
3. **Sampling errors in $\hat z$ compound across steps** — §0.3 and §0.4 together — so imagination drifts away from the manifold of real VAE codes, into a region where the RNN was never trained.

### 3.3 RSSM and PlaNet: split the state (slide 27)

*Hafner, Lillicrap, Fischer, Villegas, Ha, Lee, Davidson — "Learning Latent Dynamics for Planning from Pixels" (ICML 2019), [arXiv:1811.04551](https://arxiv.org/abs/1811.04551).*

The **Recurrent State-Space Model** factors the latent state into two parts with different jobs:

$$
\boxed{\;s_t = (h_t,\, z_t)\;}
$$

- $h_t$ — **deterministic path**. Carries memory. Computed, never sampled.
- $z_t$ — **stochastic path**. Captures uncertainty about the current state. Sampled.

**Equations.**

$$
\begin{aligned}
\text{recurrence (deterministic):}\quad & h_t = f(h_{t-1},\, z_{t-1},\, a_{t-1}) \\
\text{prior — "imagination":}\quad & p(z_t \mid h_t) \\
\text{posterior — "training":}\quad & q(z_t \mid h_t, o_t) \\
\text{observation decoder:}\quad & p(o_t \mid h_t, z_t) \\
\text{reward head:}\quad & p(r_t \mid h_t, z_t)
\end{aligned}
$$

**Fix 1 — exposure bias.** $h_t$ is deterministic: never sampled, never corrupted. Even if $z_t$ drifts, $h_t$ remains a clean memory anchor. Errors cannot compound *through* $h_t$ by resampling, because $h$ is recomputed exactly at every step from its inputs. And because $o_t$ and $r_t$ are decoded from $(h_t, z_t)$ jointly, $z_t$ is forced to encode whatever $h_t$ does not already contain.

**Fix 2 — missing prior.** The prior $p(z_t \mid h_t)$ is *learned* — no observation needed. In imagination you sample $z_t \sim p(z_t \mid h_t)$; the model has an internal compass grounded in its own memory. The KL term forces the prior to match the posterior, so imagination stays close to what training taught the model.

**The ELBO.** The generative model over a trajectory, given actions, is

$$
p(o_{1:T}, z_{1:T} \mid a_{1:T}) \;=\; \prod_{t} p(z_t \mid h_t)\, p(o_t \mid h_t, z_t)
$$

with $h_t$ a deterministic function of the past. Introduce the variational posterior $q(z_{1:T}\mid o_{1:T}, a_{1:T}) = \prod_t q(z_t \mid h_t, o_t)$ (a *filtering* posterior — it uses observations up to $t$ only). Then by Jensen:

$$
\log p(o_{1:T} \mid a_{1:T}) \;\ge\; \sum_{t=1}^{T} \Big(
\underbrace{\mathbb{E}_{q}\big[\log p(o_t \mid h_t, z_t)\big]}_{\text{reconstruction}}
+ \underbrace{\mathbb{E}_{q}\big[\log p(r_t \mid h_t, z_t)\big]}_{\text{reward}}
- \underbrace{\mathbb{E}_{q}\big[ D_{\mathrm{KL}}\big( q(z_t \mid h_t, o_t) \,\|\, p(z_t\mid h_t) \big)\big]}_{\text{prior-matching}}
\Big)
$$

Read the KL term as the whole point of the architecture: **it is the loss that makes imagination possible.** It penalises the model whenever what it *believes after seeing the observation* differs from what it *would have guessed from memory alone*. Drive it to zero and open-loop rollout equals closed-loop filtering.

**Latent overshooting.** PlaNet adds multi-step prior matching so the model is explicitly trained on the distribution it will face in imagination:

$$
\mathcal{L}_{\text{overshoot}} \;=\; \sum_{t}\sum_{d=1}^{D} \beta_d\; \mathbb{E}\Big[ D_{\mathrm{KL}}\big( \operatorname{sg}\!\big[q(z_t \mid o_{\le t})\big] \,\big\|\, p(z_t \mid z_{t-d}, a_{t-d:t-1}) \big) \Big]
$$

with $\operatorname{sg}[\cdot]$ the stop-gradient. This is a direct attack on §0.4: train the $d$-step prior to match the 1-step posterior for $d = 1 \ldots D$.

**Planning in PlaNet.** No policy at all — plain CEM in latent space at every step. Headline result: **~200× more sample-efficient than model-free methods** on DeepMind Control Suite.

### 3.4 RSSM closed-loop inference and imagination (slide 28)

At each imagined step:

1. Policy reads $(h_t, z_t)$ and outputs $a_t = \pi(h_t, z_t)$ — **no real robot needed**.
2. World model advances: $h_{t+1} = f(h_t, z_t, a_t)$, then $z_{t+1} \sim p(z_{t+1} \mid h_{t+1})$.
3. Reward head emits $\hat r_{t+1}$.
4. Accumulate over the rollout ⇒ the policy is optimised on imagined reward.

Note what is *absent*: the encoder and the decoder. In imagination, $\mathrm{enc}$ is not called (no observations) and $\mathrm{dec}$ is not called (nobody needs pixels). The decoder exists **only** to shape the representation during training. Hold that thought until Part 5, where the question becomes: if you never call the decoder at decision time, why train one at all?

### 3.5 Dreamer V1: amortise the planning (slide 29)

*Hafner, Lillicrap, Ba, Norouzi — "Dream to Control: Learning Behaviors by Latent Imagination" (ICLR 2020), [arXiv:1912.01603](https://arxiv.org/abs/1912.01603).*

> **The question:** why plan from scratch with CEM at every step when you could amortise that planning into a learned policy?

| PlaNet (2019) | Dreamer V1 (2020) |
| --- | --- |
| RSSM world model only | RSSM world model **+ actor-critic** |
| Plan at test time with CEM | Policy trained by backprop through imagined rollouts |
| ✅ actions chosen without policy bias | ✅ backprop through differentiable dynamics enables long-horizon credit assignment |
| ❌ planning from scratch at every step is expensive | ❌ policy can exploit world-model errors |

**Two alternating phases:**

1. Collect real data → train the RSSM by the ELBO (§3.3).
2. **Freeze the RSSM** → train the actor-critic purely in imagination by backprop.

**The actor and critic.**

$$
\text{actor: } a_t = \pi_\psi(h_t, z_t), \qquad \text{critic: } v_\xi(h_t, z_t) \approx \mathbb{E}_\pi\Big[\textstyle\sum_{k\ge t} \gamma^{k-t} \hat r_k\Big]
$$

**The objective: $\lambda$-returns over an imagined horizon $H$** (typically $H=15$). Define recursively backwards from the horizon:

$$
V^\lambda_t \;=\;
\begin{cases}
\hat r_t + \gamma \Big[ (1-\lambda)\, v_\xi(\hat s_{t+1}) \;+\; \lambda\, V^\lambda_{t+1} \Big] & t < H \\[4pt]
v_\xi(\hat s_H) & t = H
\end{cases}
$$

The $\lambda$ knob trades bias against variance: $\lambda \to 0$ is a one-step TD target (low variance, biased by the critic), $\lambda \to 1$ is the full imagined Monte-Carlo return (unbiased w.r.t. the *model*, high variance and maximally exposed to model error). Dreamer uses $\lambda = 0.95$.

**Actor loss** — maximise imagined returns:

$$
\mathcal{L}(\psi) \;=\; -\,\mathbb{E}_{\pi_\psi,\, p_\theta}\Big[ \sum_{t=1}^{H} V^\lambda_t \Big]
$$

**Critic loss** — regress the $\lambda$-return:

$$
\mathcal{L}(\xi) \;=\; \mathbb{E}\Big[ \sum_{t=1}^{H} \tfrac{1}{2}\big\| v_\xi(\hat s_t) - \operatorname{sg}\big[V^\lambda_t\big] \big\|^2 \Big]
$$

**The key mechanism: analytic gradients through the dynamics.** Because $z_{t+1}$ is sampled by reparameterisation ($z = \mu_\theta + \sigma_\theta \odot \epsilon$) and $a_t = \tanh(\mu_\psi + \sigma_\psi \odot \epsilon')$ likewise, the entire imagined rollout is a differentiable computation graph from $\psi$ to $V^\lambda_t$. So

$$
\nabla_\psi V^\lambda_t
$$

can be computed by ordinary backpropagation through $H$ steps of the frozen world model, rather than estimated by REINFORCE. This is a **far lower-variance gradient**: it uses the model's Jacobian $\partial \hat s_{t+1}/\partial a_t$ — actual knowledge of *how* an action changes the future — where a score-function estimator only correlates action samples with returns.

That single change is why Dreamer V1 replaced CMA-ES and outperformed model-free RL on DMControl.

**Cost of the change.** The gradient now flows through the model's errors. If $\partial \hat s_{t+1}/\partial a_t$ is wrong in a systematic direction, the actor will happily follow it into a region of the latent space that has no real-world referent. That is the ❌ in the table, and it is the price of amortisation.

### 3.6 Dreamer V2: categorical latents and KL balancing (slide 32)

*Hafner, Lillicrap, Norouzi, Ba — "Mastering Atari with Discrete World Models" (ICLR 2021), [arXiv:2010.02193](https://arxiv.org/abs/2010.02193).*

Two changes, both about the shape of the latent distribution.

**1. Categorical latents.** Replace the diagonal Gaussian $z_t$ with **32 categorical variables of 32 classes each** (a $32\times32$ one-hot matrix). Sampling is discrete; gradients use the **straight-through estimator**:

$$
z \;=\; \operatorname{onehot}\!\big(\text{sample}\big) \;+\; \operatorname{probs} \;-\; \operatorname{sg}\big[\operatorname{probs}\big]
$$

Forward pass sees the one-hot; backward pass sees the gradient of the probabilities. (See [§A.3](#a3-straight-through-and-gumbel-softmax).)

**Why discrete helps** — three arguments, all worth airing in the session:

- **Multimodality for free.** A categorical distribution over 32 classes represents 32 distinct hypotheses natively. A diagonal Gaussian cannot represent "either A or B" without also assigning mass to everything between.
- **Sparse, better-conditioned gradients.** One-hot vectors have bounded norm and don't suffer the scale pathologies of unbounded Gaussians.
- **It matches the domain.** Atari (and much of manipulation) is full of genuinely discrete events: a life is lost or not; contact is made or not. A Gaussian must smear these.

**2. KL balancing.** The KL term $D_{\mathrm{KL}}(q \| p)$ has two jobs pulling in opposite directions: train the *prior* to predict the *posterior* (good — that is dynamics learning), and pull the *posterior* toward the *prior* (dangerous — that is the posterior giving up information to make the dynamics model's life easy). Untuned, the posterior collapses toward the prior and the representation goes empty.

The fix is to weight the two directions separately using stop-gradients:

$$
\mathcal{L}_{\mathrm{KL}} \;=\; \alpha\, D_{\mathrm{KL}}\big( \operatorname{sg}[q(z_t\mid h_t,o_t)] \,\big\|\, p(z_t\mid h_t) \big) \;+\; (1-\alpha)\, D_{\mathrm{KL}}\big( q(z_t\mid h_t,o_t) \,\big\|\, \operatorname{sg}[p(z_t\mid h_t)] \big)
$$

with $\alpha = 0.8$: **80% of the pressure is on the prior to catch up with the posterior**, only 20% on the posterior to stay simple. Plus **free bits** — clip the KL below a floor so the model isn't penalised for using its first nat of information:

$$
\mathcal{L}_{\mathrm{KL}} \leftarrow \max\big(1\ \text{nat},\; \mathcal{L}_{\mathrm{KL}}\big)
$$

**Headline result.** First world-model agent to match DQN on Atari; human-level on 45 of 55 games.

### 3.7 Dreamer V3: make it work everywhere without tuning

*Hafner, Pasukonis, Ba, Lillicrap — "Mastering Diverse Domains through World Models" (2023 / Nature 2025), [arXiv:2301.04104](https://arxiv.org/abs/2301.04104).*

DreamerV3's contribution is not a new idea about dynamics; it is a set of **scale- and domain-invariance transforms** that let *one fixed hyperparameter configuration* work across Atari, DMControl, Minecraft, and robotics. That sounds like engineering, but each transform is a small piece of principled statistics, and collectively they are why V3 is the default baseline.

**1. Symlog transform.** Rewards and observations vary over many orders of magnitude across domains. Squash them:

$$
\operatorname{symlog}(x) = \operatorname{sign}(x)\,\ln\big(1 + |x|\big), \qquad
\operatorname{symexp}(x) = \operatorname{sign}(x)\,\big(\exp(|x|) - 1\big)
$$

Predictors are trained to regress $\operatorname{symlog}(y)$ and predictions are mapped back with $\operatorname{symexp}$. Unlike a plain $\log$, symlog is defined on all of $\mathbb{R}$, is smooth at 0, and is approximately the identity for small $|x|$ — so it compresses large magnitudes without distorting the small ones that matter for fine control.

**2. Twohot encoded regression.** Reward and value are predicted not by MSE regression but as a **categorical distribution over $K$ exponentially spaced bins** $b_1 < \cdots < b_K$. A scalar target $y$ is encoded onto the two adjacent bins that bracket it:

$$
\text{twohot}(y)_i \;=\;
\begin{cases}
\dfrac{b_{k+1} - y}{b_{k+1} - b_k} & i = k \\[8pt]
\dfrac{y - b_{k}}{b_{k+1} - b_k} & i = k+1 \\[6pt]
0 & \text{otherwise}
\end{cases}
\qquad \text{where } b_k \le y \le b_{k+1}
$$

trained with categorical cross-entropy, and read out as $\hat y = \operatorname{symexp}\big(\sum_i \operatorname{softmax}(\ell)_i\, b_i\big)$. Two benefits: the gradient magnitude no longer scales with the error magnitude (so a single rare huge reward cannot blow up training), and the head can represent a *multimodal* value distribution.

**3. Return normalisation.** Scale advantages by a robust range statistic rather than a standard deviation:

$$
S \;=\; \operatorname{Percentile}\big(V^\lambda,\, 95\big) - \operatorname{Percentile}\big(V^\lambda,\, 5\big), \qquad
\text{advantage} \;\leftarrow\; \frac{V^\lambda_t - v_\xi(\hat s_t)}{\max(1,\, S)}
$$

The percentile range ignores outliers; the $\max(1, \cdot)$ prevents *amplifying* noise in sparse-reward settings where returns are almost always zero (a plain standard-deviation normaliser would divide by ~0 and explode).

**4. Actor loss** combines a policy-gradient term on the normalised advantage with an entropy bonus:

$$
\mathcal{L}(\psi) \;=\; -\sum_{t}\Big( \operatorname{sg}\Big[\tfrac{V^\lambda_t - v_\xi(\hat s_t)}{\max(1,S)}\Big]\, \log \pi_\psi(a_t \mid \hat s_t) \;+\; \eta\, \mathcal{H}\big[\pi_\psi(\cdot \mid \hat s_t)\big] \Big)
$$

(V1's analytic backprop-through-dynamics remains available for continuous control; the released configurations differ by domain — check the code before asserting which is used where.)

**5. Free bits + KL balancing retained**, with separate coefficients for the dynamics ("prior catches up") and representation ("posterior stays simple") directions.

**Headline result.** **First method to collect diamonds in Minecraft from scratch, from pixels, with no human data and no curriculum** — a task requiring roughly 20,000 sequential decisions with extremely sparse reward — using the same hyperparameters as on Atari and DMControl.

**Why this matters for a study group.** V3 is the strongest available argument that world-model RL is *usable*, not just publishable. The robustness transforms are the difference between "works after a two-week hyperparameter sweep per environment" and "works." When you do the build milestone, steal symlog and twohot first.

### 3.8 Dreamer V4: decoupling video from actions (slide 33)

*Hafner et al. — "Training Agents Inside of Scalable World Models" (2025).*

The bottleneck for V1–V3 is that the world model must be trained on **action-labelled** data from the target embodiment. V4 breaks that.

**Two-stage training:**

$$
\underbrace{p(\hat z_{t+1} \mid z_{\le t})}_{\text{pretrain on unlabelled video}} \;\longrightarrow\; \underbrace{p(z_{t+1} \mid z_{\le t},\, a_{\le t})}_{\text{fine-tune with actions}}
$$

Stage 1 learns "how the world evolves" from action-free video. Stage 2 learns "how *my* actions perturb that evolution" from a much smaller labelled set. Since the action-free stage carries the vast majority of the visual-dynamics burden, this is the same economics argument that motivates video backbones in Part 4 — arrived at from the RL side.

**Two architectural changes:**

1. **Shortcut forcing / flow matching** replaces the categorical-latent + KL machinery for the generative part. Flow matching (see [§A.6](#a6-flow-matching-and-shortcut-models)) trains a velocity field to transport noise to data; "shortcut" variants additionally condition on the step size so that few-step (even one-step) sampling is trained directly rather than distilled afterwards. Combined with teacher forcing along the time axis, this gives a model that can be rolled out fast enough for RL. The general name for the time-axis variant in the literature is *diffusion forcing*: independent noise levels per frame, so the model learns to denoise the future while conditioning on clean past.
2. **Transformer KV-cache replaces the RNN hidden state as memory.** $h_t$ becomes the attention cache over past tokens rather than a fixed-width recurrent vector. Consequences: memory capacity scales with context length instead of hidden width; training parallelises over time; and long-range dependencies (an item picked up 500 steps ago) survive. Cost: memory grows with context, and you inherit the whole context-length engineering problem.

**Headline results.** Diamonds in Minecraft **from offline data only** — 20,000+ decisions with no environment interaction — plus real-time operation on a single GPU and learning from video.

**Lineage summary (slide 32):**

| Year | Model | Key innovation | Headline result |
| --- | --- | --- | --- |
| 2018 | World Models | V–M–C decomposition (VAE + MDN-RNN + CMA-ES) | First agent trained purely in imagination; Car Racing from pixels |
| 2019 | PlaNet | RSSM: deterministic $h_t$ + stochastic $z_t$ | 200× more sample-efficient than model-free; CEM planning in latent space |
| 2020 | Dreamer V1 | Backprop through dynamics; actor-critic replaces CMA-ES | Outperforms model-free RL on DMControl; policy learned entirely in imagination |
| 2021 | Dreamer V2 | Categorical latents + KL balancing | First world model to match DQN on Atari; human-level on 45/55 games |
| 2023 | Dreamer V3 | Symlog + twohot + return normalisation; fixed hyperparameters | First to obtain diamonds in Minecraft from pixels; same hyperparameters across domains |
| 2025 | Dreamer V4 | Flow matching / shortcut forcing + transformer KV-cache; unlabelled video pretraining | Diamonds from offline data only; real-time on a single GPU |

*Worth mentioning in passing:* **IRIS** (Micheli et al., 2023) took the discrete-token route — a VQ-VAE tokeniser plus an autoregressive transformer world model — reaching superhuman Atari 100k performance, and is the direct ancestor of the tokenised video world models in Part 4. **TD-MPC2** (Hansen et al., 2024) is the strongest counterpoint in the other direction: a latent model trained with **no reconstruction at all**, only reward/value/dynamics-consistency losses, combined with MPPI planning — evidence that the decoder is optional even inside Family II.

### 3.9 DayDreamer: does it survive contact with hardware? (slide 30)

*Wu\*, Escontrela\*, Hafner\* et al. — "DayDreamer: World Models for Physical Robot Learning" (CoRL 2022).*

Dreamer on real robots, **with the same hyperparameters**, learning online without a simulator:

| Robot | Task | Wall-clock training |
| --- | --- | --- |
| A1 quadruped | Walking from scratch | ~1 hour |
| UR5 | Multi-object visual pick-and-place | ~8 hours |
| XArm | Visual pick-and-place | ~10 hours |
| Sphero | Ollie visual navigation | ~2 hours |

**Why it works:** one hour of real quadruped interaction becomes *days* of imagination training. The replay buffer is small and the model is trained hard on it; the policy gets orders of magnitude more gradient steps than real steps. This is the sample-efficiency argument cashed out in wall-clock time, and it is the single most convincing demonstration that latent AC-WMs are practical for robots.

**Caveats for discussion.** The tasks are dense-reward and short-horizon; reward has to be engineered or instrumented in the real world (a non-trivial problem the paper solves per-task); and the robot must survive its own exploration.

### 3.10 The Matrix Dojo (slide 31)

The slide is a joke, but it is the right joke. "Training in a dream" only works to the extent that the dream is faithful *where the policy chooses to go*. Because the policy is *optimising against the model*, it actively seeks the model's blind spots. Standard mitigations:

- **Uncertainty penalties**: subtract $\beta \cdot u(\hat s_t)$ from imagined reward, where $u$ is ensemble disagreement or a learned epistemic estimate (MOPO/MOReL-style pessimism).
- **Short horizons**: $H = 15$ rather than 500 (§0.3).
- **Frequent real data**: alternate model training and policy training so the model keeps being refit on the policy's own state distribution — the model-based analogue of DAgger.
- **Noisier dreams**: Ha & Schmidhuber's temperature trick (§3.1.2).

---

## Part 4 — Families III & IV: generative video as a world model

> **The pivot question (slide 34):** *Dreamer's latents are domain-specific. What if we want to scale the videos to all of the Internet?*

A Dreamer latent is learned from scratch on one environment. It cannot be pretrained on YouTube, because YouTube has no actions and no rewards, and the latent has no meaning outside its training domain. Meanwhile there exist ~$10^9$ hours of video showing physical interaction. The question of Part 4 is how to convert that into robot competence.

### 4.1 The economic argument: VLAs are sample-inefficient (slides 39–41)

**The VLA pipeline:**

```text
image–text pairs  →  VLM  →  [ large-scale robotics data ]  →  VLA
   (semantics)                (learn: dynamics + control)
                                ❌ expensive post-training
```

A VLM trained on static images and text is, as slide 11 put it, *blind to physical causality and temporal dynamics*. Turning it into a VLA therefore requires the teleoperation data to teach **both** dynamics and control — and teleoperation data is the scarcest resource in robotics.

**The video-backbone pipeline:**

```text
video–text pairs  →  video model  →  [ small-scale robotics data ]  →  VAM
(semantics + visual dynamics)              (learn: control only)
                                         ✅ efficient post-training
```

Offload dynamics learning to **action-free video**, which is abundant and free. Then robot data only has to bridge from "what should happen" to "what my joints should do." The claimed payoff (slide 41) is **~10× sample efficiency**: a VAM reaches the success rate that a VLA needs 100% of the robot dataset for, using around 10%.

**The strong empirical claim (slide 42).** Policy performance scales with the quality of the video model: a *finetuned* video model beats a *pretrained* one at the same action-decoder budget, and predicted-video-conditioned action decoding approaches expert-video-conditioned decoding. In other words, the video model is not a fancy feature extractor — improving its *prediction* improves *control*. That is the empirical core of the whole family and the thing to interrogate.

### 4.2 The token economics problem (slides 35–36)

Naive per-frame ViT tokenisation:

$$
\underbrace{\frac{256 \times 256}{16 \times 16}}_{\text{patches per frame}} = 256 \ \text{tokens/frame}
\quad\times\quad
\underbrace{20\ \text{FPS} \times 10\ \text{s}}_{200\ \text{frames}}
\;=\; \mathbf{51{,}200\ \text{tokens}} \ \text{for a 10-second clip}
$$

That fills a large fraction of a 128K context with *one short clip*. Attention cost is $\mathcal{O}(L^2)$, so this is not a problem you spend your way out of. Compression is mandatory, and it comes on three independent, composable axes.

#### Axis 1 — spatial compression

Deeper encoder ⇒ fewer tokens per frame.

- **Mechanism:** VQ-VAE / VQGAN encoder with stride. $256\times256 \to 4\times4$ tokens = **64× spatial reduction**, i.e. 256 → 4 tokens/frame.
- **Examples:** VQGAN, SD-VAE, Cosmos CV8×8×8.
- **Causal?** Yes — no temporal dependency at all, each frame is encoded independently.

#### Axis 2 — temporal compression

Merge frames ⇒ fewer time steps. Two mechanisms with different causality properties:

- **Tubelets.** A 3D patch $(t,h,w)$ encodes a whole tube at once. **Non-causal** — the token for time $t$ depends on frames after $t$. Fine for offline video understanding (TimeSformer, ViViT, Cosmos analysis), fatal for interactive rollout.
- **Causal aggregation.** Attend to past only; decode frame by frame. **Causal** — usable in a world model. (Dreamer V4's tokeniser, Cosmos world model.)

Typical: 200 → 25 time steps = **8× temporal reduction**.

> **The distinction that matters for this lecture.** If your tokeniser is non-causal, you cannot use it in a world model, because you would need the future to encode the present. Whenever you read a video-model paper, check this first. *The causality requirement determines which temporal method you may use.*

#### Axis 3 — adaptive compression

Variable token budget — complex frames get more tokens.

- **Mechanism:** tail-drop masking during training; tokens ordered by information content so a prefix is a valid lower-fidelity encoding. At inference, spend tokens proportionally to frame complexity.
- **Example:** ElasticTok (Yan et al., ICLR 2025).
- **Causal?** Yes — works per-frame with context.
- Typical: **2–5× further reduction on average.**

**Composing them (the Cosmos numbers):** spatial 8× **×** temporal 8× **=** 512× total ⇒ **51,200 tokens becomes ~100 tokens for a 10-second clip.**

### 4.3 Where do actions live? (slide 37)

This is the taxonomic heart of Part 4. Three answers.

**A. Action-Conditioned World Models (AC-WMs) — actions in, future states out.**

$$
[\text{observations} + \text{actions}] \;\longrightarrow\; p(s_{t+1} \mid s_t, a_t) \;\longrightarrow\; [s_{t+1}]
$$

Given current observations and planned actions, simulate what the world will look like. $a_t$ is an **input** that conditions the prediction. *Examples: Dreamer V1–V4, DreamDojo, V-JEPA 2.*

**B. WAM — World Action Model.** A *single* model jointly predicts video **and** actions:

$$
p\big(o_{1:t+H},\; a_{t:t+H} \;\big|\; \text{text},\, o_{0:t},\, q_t \big)
$$

($q_t$ = proprioception.) The action is an **output**, generated in the same forward pass as the video, sharing the same denoising/decoding trajectory. *Example: DreamZero.*

**C. VAM — Video Action Model.** A **frozen** video backbone pretrained on internet video produces video features; a lightweight head (an inverse dynamics model, trained on robot data only) reads actions off them:

$$
\text{image} + \text{text} \;\to\; \underbrace{\text{video backbone (frozen)}}_{\text{internet video}} \;\to\; \text{video features} \;\to\; \underbrace{\text{IDM}}_{\text{robot data only}} \;\to\; \text{actions}
$$

*Example: mimic-video.*

**The conceptual difference between B and C** is where the robot-specific knowledge lives. In a VAM it is quarantined in a small head, so the video backbone keeps its general visual competence and can be shared across embodiments. In a WAM it is distributed through the whole model, which allows tighter video–action coupling at the cost of specialising the backbone.

### 4.4 mimic-video: the VAM instantiation (slides 43–46)

*Pai\*, Achenbach\*, Montesinos, Forrai, Mees\*, Nava\* — "mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs" (arXiv, 2025).*

**Architecture (slide 43).** A language encoder embeds the instruction ("put the package on the conveyor belt"); a video diffusion model denoises a future video latent conditioned on the current frame and the language embedding; an action head reads the (predicted) video representation and emits an action chunk. Repeat.

**Results (slides 44–45).**

- State-of-the-art on LIBERO-Object / Spatial / Goal, SIMPLER-Bridge, real-world dexterous bimanual manipulation.
- **10× more sample-efficient** than a $\pi_0$-style VLA (matching its success rate at ~10% of the robot data).
- **Trains ~2× faster** in wall-clock terms at equal performance.

**Joint video–action sampling (slide 46).** At deployment you can *skip the video generation branch* and run only the action head. This is the practical resolution of the "video is expensive" objection: the video model is a **training-time and debugging-time** device. During regular policy inference you pay only the action head. When something goes wrong, you turn the video branch back on and *watch what the policy thinks is about to happen* — which is an interpretability affordance no VLA offers.

**The question to press on.** If you can skip video generation at inference, in what sense is the video model a *world model* rather than a very good pretrained representation? The scaling evidence on slide 42 (finetuning the video model improves control) is the argument that prediction is doing real work — is that evidence sufficient?

### 4.5 DreamZero: the WAM instantiation (slides 47–48)

*Ye et al. — "World Action Models are Zero-shot Policies" (arXiv, 2026).*

**Claim:** joint video–action prediction, **no explicit IDM**; autoregressive video-chunk generation with KV-cache inference on a 14B model.

**Training — joint video-action flow matching.**

```text
Video  --[ VAE encoder ]--> latents --+
                                      |--(+noise)--> [ Causal DiT blocks ] --> joint flow matching
Action --[ action encoder ]-----------+                                        (teacher forcing)
Proprio + Language --[ state/text encoder ]-------^
```

Both the video latents and the action chunk are noised and denoised **in the same DiT**, with a single joint flow-matching objective. Writing $x = (x^{\text{vid}}, x^{\text{act}})$ for the concatenated clean target, $\epsilon \sim \mathcal{N}(0,I)$, and $\tau \in [0,1]$:

$$
x_\tau \;=\; (1-\tau)\,\epsilon \;+\; \tau\, x, \qquad
\mathcal{L} \;=\; \mathbb{E}_{x,\epsilon,\tau}\Big[\big\| v_\theta\big(x_\tau,\, \tau,\, c\big) \;-\; (x - \epsilon) \big\|_2^2\Big]
$$

with conditioning $c = (\text{past frames},\ \text{proprioception},\ \text{language})$. See [§A.6](#a6-flow-matching-and-shortcut-models).

**Why "no explicit IDM" is the point.** A VAM predicts video, then a *separate* module infers "what action would have produced that transition." That is a two-stage pipeline with an information bottleneck: the IDM sees only the video, so any action detail not visible in pixels (force, stiffness, grip aperture below pixel resolution) is unrecoverable. A WAM lets action tokens attend to video tokens *and vice versa* throughout the network, so actions are predicted with full access to the model's internal state — and, symmetrically, the video prediction is informed by the action being generated.

**Inference — closed-loop real-world execution.** Autoregressive flow sampling with a KV cache produces future frames *and* a future action chunk; the action chunk executes asynchronously on the real robot while the next chunk is generated; real observations are folded back in ("update with real observation"). This is MPC with a learned amortised proposal instead of a sampling-based optimiser.

**Demonstrated tasks (slide 48).** Long-horizon deformable manipulation: multi-step shirt folding from a paragraph-long instruction, untying a shoelace, picking up a marker and drawing a circle on a book. These are exactly the tasks where hand-engineered simulators are useless and where a video prior should pay off.

### 4.6 WAMs versus AC-WMs (slide 49)

| Dimension | WAMs (DreamZero, mimic-video) | AC-WMs (Dreamer, DreamDojo) |
| --- | --- | --- |
| **Data** | ➖ instruction-labelled only — hard to use play data or failure trajectories at scale | ➕ all robot data: play, failures, policy rollouts — easier to scale |
| **Cross-embodiment** | ➕ easy — inputs/outputs embodiment-agnostic; only the action decoder needs robot data | ➖ open challenge — each robot has its own action space |
| **Pre-training preservation** | ➕ small gap from the pretrained video model; preserves general visual capabilities | ➖ action conditioning changes the input distribution; may destroy pretrained abilities |
| **Beyond imitation** | ➖ BC paradigm only — instructions → actions; no RL, no counterfactual simulation | ➕ RL in imagination + fine-grained gradient-based planning |
| **Planning** | ➕ best-of-$N$ via diverse text prompts; easy action-proposal generation | ➕ fine-grained gradient-based optimisation of action sequences at inference |

**The trade-off in one line.** WAMs inherit the internet; AC-WMs inherit counterfactuals. A WAM can tell you what a competent agent would probably do; an AC-WM can tell you what *would happen* if you did something no one has ever done. The second is what you need for genuine planning, RL, and safety analysis — and it is precisely what internet video cannot teach you, because internet video contains no counterfactuals.

**"Beyond imitation" is the row to argue about.** It is the strongest reason to think WAMs are a way-station rather than the destination.

---

## Part 5 — Family V: predict representations, not pixels (JEPA)

> **The pivot question (slide 50):** *All approaches so far had a decoder to reconstruct or generate pixels at some point. Do you even need to predict pixels at all?*

Recall §3.4: in Dreamer's imagination loop the decoder is never called. It exists only to shape the representation during training. So the question is sharp — is the decoder a *useful auxiliary objective* or a *tax*?

**The case that it is a tax.** Reconstruction spends capacity in proportion to pixel variance. A waving tree, a shifting shadow, TV static, wood grain — all expensive to reconstruct, all irrelevant to whether the grasp succeeds. Worse, they are *unpredictable*, so a model forced to predict them wastes capacity on irreducible noise. The canonical thought experiment: a Dreamer agent facing a screen of TV static will devote most of its ELBO to modelling the static.

### 5.1 The JEPA objective (slide 51)

Predict **in latent space** — no decoder, no pixel reconstruction:

$$
z_t = \mathrm{enc}(o_t), \qquad
\hat z_{t+1} = \mathrm{pred}(z_t, a_t), \qquad
z_{t+1} = \mathrm{enc}(o_{t+1})
$$

$$
\mathcal{L}_{\text{JEPA}} \;=\; \big\| \hat z_{t+1} - z_{t+1} \big\|_2^2
$$

Elegant. Also, as written, **broken**.

### 5.2 Representation collapse

The trivial minimiser: let $\mathrm{enc}(\cdot) \equiv c$ for a constant $c$, and $\mathrm{pred}(\cdot, \cdot) \equiv c$. Then $\mathcal{L} = 0$ exactly, for every input, and the representation carries zero information.

This is not a hypothetical — it is the *attractor* of naive gradient descent on this loss, because reducing the variance of the encoder output reduces the loss faster than improving the predictor does. Reconstruction-based objectives are immune (a constant latent cannot reconstruct the image); a purely predictive latent objective is not. **Every JEPA-family method is, at bottom, an anti-collapse mechanism plus this loss.**

### 5.3 Three anti-collapse strategies (slide 51)

| Strategy | Mechanism | Trade-off | Example |
| --- | --- | --- | --- |
| **EMA target encoder** | Target encoder is a slow-moving momentum copy of the online encoder; no gradient flows through it — this breaks the gradient symmetry that makes collapse a descent direction | Not fully end-to-end; requires tuning the momentum schedule; no proof of non-collapse | I-JEPA, V-JEPA 2 |
| **Frozen pretrained encoder** | Encoder is fixed (e.g. DINOv2). Collapse is *impossible* because the encoder never changes | No end-to-end learning; you are stuck with whatever the pretrained features capture | DINO-WM |
| **Gaussian regularisation** | Force the latent distribution to be isotropic Gaussian; provably prevents collapse | Fully end-to-end, one hyperparameter | LeJEPA / LeWorldModel |

**On EMA.** Write $\theta$ for the online encoder and $\bar\theta$ for the target, with $\bar\theta \leftarrow m\bar\theta + (1-m)\theta$, $m \approx 0.996$. The loss is $\|\mathrm{pred}_\phi(\mathrm{enc}_\theta(o_t), a_t) - \mathrm{enc}_{\bar\theta}(o_{t+1})\|^2$ with a stop-gradient on the target. Collapse requires *both* encoders to shrink together; the momentum lag means the online encoder is always chasing a stale target, and the predictor can reduce the loss faster by actually modelling the transition than by shrinking. It works, reliably, and there is still no fully satisfying theory of *why* — which is a nice open problem to put in front of a study group.

**On explicit regularisation.** The VICReg family adds terms that directly forbid collapse — a hinge on the per-dimension standard deviation and an off-diagonal covariance penalty:

$$
v(Z) = \frac{1}{d}\sum_{j=1}^{d} \max\big(0,\; \gamma - \sqrt{\operatorname{Var}(z_{\cdot j}) + \epsilon}\big),
\qquad
c(Z) = \frac{1}{d}\sum_{i \ne j} \big[C(Z)\big]_{ij}^2
$$

The variance term makes each dimension refuse to be constant; the covariance term makes dimensions refuse to be redundant. LeJEPA's contribution is to replace this pile of heuristics with a single distributional target — push the embedding distribution toward an isotropic Gaussian (via a sketched test over random projections) — with a proof that this prevents collapse and one hyperparameter to set.

### 5.4 🔶 DINO-WM: the frozen-encoder instantiation

*Zhou, Pan, LeCun, Pinto — "DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning", [arXiv:2411.04983](https://arxiv.org/abs/2411.04983).*

The cleanest possible test of "do you need pixels": take a **frozen DINOv2**, learn only dynamics on top, and plan.

**Observation model.** DINOv2, frozen during training *and* testing, encodes $o_t$ to **patch embeddings**

$$
z_t \in \mathbb{R}^{N \times E}
$$

($N$ patches, $E$ dims). Note this is a *spatial* latent, not a single global vector — patches preserve the "where," which is what manipulation needs.

**Action and proprioception conditioning.** A $K$-dimensional action vector is produced by an MLP $\phi$ from the raw action and **concatenated to every patch vector** $z_t^i$, $i = 1,\ldots,N$. Proprioception, when available, is concatenated the same way.

**Transition model.** A ViT predictor over the patch tokens of the last $H$ frames.

**Training loss (their Eq. 1)** — pure latent consistency, teacher-forced on segments of length $H+1$:

$$
\mathcal{L}_{\text{pred}} \;=\; \Big\| \, p_\theta\big(\mathrm{enc}_\theta(o_{t-H:t}),\; \phi(a_{t-H:t})\big) \;-\; \mathrm{enc}_\theta(o_{t+1}) \Big\|^2
$$

*No pixel reconstruction anywhere in world-model training.*

**Decoder (their Eq. 2) — for interpretability only.** A stack of transposed convolutions $q_\theta$ trained with

$$
\mathcal{L}_{\text{rec}} \;=\; \big\| q_\theta(z_t) - o_t \big\|^2
$$

**trained entirely independently of the transition model.** The paper explicitly ablates backpropagating the decoder loss into the predictor and finds it **hurts** performance. That is the sharpest empirical statement of the "decoder is a tax" thesis in the literature, and it is worth putting on a slide for the session.

**Planning.** Given a current observation $o_0$ and a goal image $o_g$:

$$
\hat z_t = p(\hat z_{t-1}, a_{t-1}), \qquad \hat z_0 = \mathrm{enc}(o_0), \qquad z_g = \mathrm{enc}(o_g)
$$

$$
C \;=\; \big\| \hat z_T - z_g \big\|^2
$$

Optimise $a_{0:T}$ against $C$ inside an MPC loop with **CEM**. The paper notes the model is differentiable so gradient descent is available and cheaper, but reports that **CEM empirically outperforms GD** — consistent with the general finding that learned-dynamics loss surfaces are non-convex enough to trap first-order optimisers.

**Why "zero-shot planning" is the right phrase.** No reward function, no expert demonstrations, no learned inverse model, no task-specific training. A task is just *an image of the goal*. Evaluated across six suites — maze navigation (Maze, Wall), tabletop pushing (PushT), arm control (Reach), and deformable/granular manipulation (Rope, Granular) — with 224×224 RGB observations. The comparison against DreamerV3 and IRIS baselines is the headline: latent-only prediction on frozen general features beats reconstruction-based world models at planning.

**The ablation that carries the argument.** They swap DINOv2 for other pretrained encoders and show downstream planning performance tracks encoder quality. The world model's ceiling is set by the representation, not by the dynamics head — which is the strongest available argument for "spend your compute on representation pretraining, not on dynamics architecture."

### 5.5 🔶 V-JEPA 2 and V-JEPA 2.1: the EMA instantiation at scale

*V-JEPA 2: Assran et al. (2025). V-JEPA 2.1: Mur-Labadia, Muckley, Bar, Assran, Sinha, Rabbat, LeCun, Ballas, Bardes — "V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning", [arXiv:2603.14482](https://arxiv.org/abs/2603.14482).*

**V-JEPA 2** is the scaling test for the JEPA hypothesis: a ~1B-parameter ViT trained with a masked latent-prediction objective and an EMA target on over a million hours of internet video. Then **V-JEPA 2-AC** adds action conditioning by fine-tuning on a modest amount (order tens of hours) of unlabelled Droid robot video, and plans zero-shot with CEM in latent space for pick-and-place on a robot it was never trained to control.

The structure is exactly the Part 4 economics argument, but with a JEPA rather than a generative backbone: **action-free video for dynamics, a small action-conditioned adapter for control, and no pixels ever generated.**

**V-JEPA 2.1** addresses a known weakness — JEPA embeddings are strong globally but weak *densely* (per-patch, spatially precise), which is exactly what manipulation needs. Four components:

1. **Dense predictive loss** — both visible and masked tokens contribute, promoting spatial and temporal grounding.
2. **Deep self-supervision** — apply the objective at multiple intermediate encoder layers, not only the output.
3. **Multi-modal tokenisers** — unified training over images and videos.
4. **Scaling** — benefits from more capacity and data.

Reported: 7.71 mAP on Ego4D object-interaction anticipation, 40.8 Recall@5 on EPIC-KITCHENS action anticipation, and a **20-point improvement in robot grasping success over V-JEPA 2**, plus gains on navigation and depth estimation.

**The significance for the "can JEPA scale?" open question (slide 52).** The answer is trending yes, and the specific finding — that *dense* features rather than *global* features are the bottleneck for manipulation — is a concrete, testable claim about what robot representations need.

### 5.6 🔶⚠️ μ0: traces as the intermediate state

*[mu0-wm.github.io](https://mu0-wm.github.io/).* ⚠️ *From the project page; no full paper read.*

μ0 proposes a third option between "predict pixels" and "predict an opaque latent": predict **3D traces of semantic interaction points** — objects, tools, hands, contact regions.

$$
\underbrace{\text{video-only pretraining}}_{\text{learn what must move}} \;\longrightarrow\; \underbrace{\text{3D interaction traces}}_{\text{embodiment-agnostic}} \;\longrightarrow\; \underbrace{\text{Action Expert}}_{\text{robot-specific control}}
$$

**The argument.** Pixel-space video models waste capacity on appearance; action-labelled agents are locked to one embodiment. A trace — "this contact point moves along this 3D path" — describes *what must happen* independent of the hardware that makes it happen. It is a state representation with explicit geometric semantics, so unlike a Dreamer latent it transfers across robots, and unlike pixels it is cheap.

**Results.** RoboCasa365: **30.25% average success across 8 tasks, +5.0 points over $\pi_0$.** Real-world UR3 manipulation (pick & place, pour).

**Context.** Concurrent work (MolmoMotion, UMA, LUCID) converges on the same idea — 3D trajectories/motion as an embodiment-agnostic bridge between video and control. This is arguably the most active current answer to slide 52's "will flexible conditioning win?" question, and it deserves a slot in the discussion because it is *not* one of the deck's five families.

**The objection to raise.** Traces are a *hand-chosen* intermediate representation. The whole trajectory of this lecture has been "stop hand-engineering, let the data decide" — physics engine → learned pixel model → learned latent → learned features. Is μ0 a principled inductive bias, or the reintroduction of a bottleneck we spent a decade removing? (The honest answer is probably "it depends whether contact geometry really is the sufficient statistic for manipulation," which is an empirical question you could design an experiment around.)

### 5.7 Does the decoder ever earn its keep?

Collect the evidence on both sides before the session; this is the best debate in the lecture.

**For the decoder:**

- DIAMOND: visual details matter; discrete compression drops the sprites the agent needs.
- Interpretability: you can *watch* a rollout and debug it. mimic-video keeps the video branch for exactly this.
- Reconstruction is a dense, well-conditioned learning signal that regularises the representation early in training.
- Human interfaces: goal specification, teleoperation previews, playable models (MIRA).

**Against the decoder:**

- DINO-WM: backpropagating the decoder loss into the predictor **hurts** planning performance.
- TD-MPC2: no reconstruction at all, strong continuous control.
- Capacity spent in proportion to pixel variance, which is uncorrelated with task relevance.
- Irreducible noise (static, textures) is an unclosable loss floor that distorts the whole objective.

**Reframe that dissolves some of the disagreement.** The relevant axis may not be "decoder or not" but *"is your representation learned from a task-relevant objective or an appearance objective?"* DINOv2 features are learned from a discriminative objective on internet images, which happens to align with task relevance. Atari pixels *are* the task-relevant state. A VAE trained on the target domain is neither. Test this reframing against each paper.

---

## Part 6 — The sideways move: skip the world model

### 6.1 🔶⚠️ Pantograph PAN-1: goal-conditioned RL from action-free video

*[pantograph.com/journal/pan-1](https://pantograph.com/journal/pan-1).* ⚠️ *From the journal post; no full paper read.*

Every method so far accepts the same premise: *action-free video is valuable, so learn dynamics from it and add actions later.* PAN-1 rejects the premise. If the goal is a policy, learn the **policy** from video directly and skip the world model.

**Setup.** A 4B model trained to achieve diverse goals in Minecraft from visual frames, with **no action labels during pretraining**.

**The trick: hindsight relabelling.** Treat internet videos as RL trajectories that happen to contain only observations. Any *future frame* within a video is a valid goal for any *earlier* frame. So:

> what may have been a failure at one goal becomes a success for whatever actually happens.

Every segment of every video is optimal-by-construction for *some* goal. This converts unlabelled video into supervised goal-conditioned data at essentially zero cost — the same move as Hindsight Experience Replay, applied at internet scale.

**The math: the successor measure.** For a policy $\pi$, define the discounted state-occupancy measure

$$
M^\pi(s, \cdot) \;=\; (1-\gamma) \sum_{t \ge 0} \gamma^{\,t}\; \mathbb{P}\big(s_t \in \cdot \;\big|\; s_0 = s,\, \pi\big)
$$

— "the probability that a goal state appears in the agent's future." Now take the indicator reward $r_g(s) = \mathbb{1}[s = g]$. The value function is

$$
V^\pi_g(s) \;=\; \sum_{t\ge0}\gamma^t\,\mathbb{P}(s_t = g \mid s_0 = s, \pi) \;=\; \frac{M^\pi(s,g)}{1-\gamma}
$$

**So the successor measure *is* the goal-conditioned value function, for every goal at once.** Learning it (by likelihood-based or energy-based modelling of the goal distribution) converts a multi-step decision problem into a **single-step density-estimation problem** — no bootstrapping, no rollouts, no dynamics model.

**Pipeline.** Pretrain on ~500k hours of action-agnostic Minecraft video; post-train on ~2k hours of contractor trajectories with action labels.

**Results.** Across 104 evaluation environments, Pan-4B substantially outperforms baselines: **85.7% vs. STEVE-1's 16.5%** on basic navigation; **24.4% vs. 15.1%** for the next-best model on complex mechanisms.

**Why this belongs in a world-models lecture.** It is the sharpest available null hypothesis. The lecture's entire argument is "learn dynamics from video, then add control." PAN-1 says "learn *control* from video directly; the dynamics were only ever instrumental." The counter-argument is the "beyond imitation" row of slide 49: a goal-conditioned policy cannot answer counterfactual questions, cannot be re-planned against a new cost function, and cannot support safety analysis. But if the deliverable is a policy, that may not matter.

**Discussion prompt.** Under what conditions is the world model *necessary* rather than merely one route to a policy? Candidate answers to test: novel objectives at test time; safety verification; multi-agent reasoning; data efficiency in the low-data regime; anything requiring "what if I did the thing nobody has ever done."

---

## Part 7 — The planning toolbox

You need this to run the build milestone and to read any of the papers above properly.

### 7.1 The MPC loop

```text
repeat at every control step:
  1. encode the real observation:  s_t ← enc(o_t)
  2. optimise a candidate action sequence a_{t:t+H} against the model
  3. execute ONLY a_t   (the first action, or a short chunk)
  4. discard the rest of the plan
  5. observe the real o_{t+1}, go to 1
```

Steps 3–4 are not an inefficiency — they are the **error-control mechanism** derived in §0.3. Re-encoding the real observation resets the compounding-error clock every step. Replan frequency is the single most important hyperparameter in model-based control, and the first thing to ablate.

### 7.2 Random shooting, CEM, and MPPI

Let $S(a_{0:H})$ be the predicted cost of an action sequence under the model (negative predicted reward, or goal distance as in DINO-WM).

**Random shooting.** Sample $N$ sequences from a fixed distribution, evaluate, take the best. Trivial, embarrassingly parallel, and a surprisingly strong baseline. Use it as your control in the build milestone.

**Cross-Entropy Method.** Iteratively refit a sampling distribution to its own best samples. With a diagonal Gaussian $q_\theta = \mathcal{N}(\mu, \operatorname{diag}\sigma^2)$ over the flattened action sequence, for iteration $i = 1 \ldots I$:

1. Sample $\{a^{(k)}\}_{k=1}^{N} \sim \mathcal{N}(\mu_i, \operatorname{diag}\sigma_i^2)$.
2. Evaluate $S(a^{(k)})$ by rolling out the model.
3. Take the **elite set** $\mathcal{E}$ = the top-$k$ lowest-cost samples.
4. Refit by moment matching:

$$
\mu_{i+1} = \frac{1}{|\mathcal{E}|}\sum_{a \in \mathcal{E}} a, \qquad
\sigma^2_{i+1} = \frac{1}{|\mathcal{E}|}\sum_{a \in \mathcal{E}} (a - \mu_{i+1})^2
$$

Optionally smooth: $\mu_{i+1} \leftarrow \alpha\mu_{i+1} + (1-\alpha)\mu_i$.

*Where the name comes from.* Define a target distribution concentrated on good sequences, $q^\star(a) \propto \mathbb{1}[S(a) \le \gamma_i]\, q_{\theta_i}(a)$. The update minimises the cross-entropy $D_{\mathrm{KL}}(q^\star \| q_\theta)$ over the Gaussian family, and for an exponential family that minimisation *is* moment matching to the elite samples. So CEM is not a heuristic — it is exact variational inference over an indicator-reweighted proposal.

**MPPI** (Model-Predictive Path Integral) is the soft version: instead of a hard top-$k$ cut, weight every sample by an exponentiated cost.

$$
w_k \;=\; \frac{\exp\!\big(-\tfrac{1}{\beta}\, S(a^{(k)})\big)}{\sum_{j=1}^{N} \exp\!\big(-\tfrac{1}{\beta}\, S(a^{(j)})\big)},
\qquad
\mu \;\leftarrow\; \sum_{k=1}^{N} w_k\, a^{(k)}
$$

$\beta$ is a temperature: $\beta \to 0$ recovers "take the single best sample"; $\beta \to \infty$ recovers "ignore the costs." MPPI uses all samples (lower variance than a hard cut) and has a derivation from the path-integral control / KL-control literature, where the exponential weighting is the exact solution to a KL-regularised control problem.

**Practical notes.** Correlate noise across time (sample smooth action sequences, e.g. via a coloured-noise or spline parameterisation) — white noise per timestep produces jittery sequences that no real actuator tracks and that the model has never seen. Warm-start $\mu$ from the previous step's plan, shifted by one.

### 7.3 Gradient-based planning

If the model is differentiable — and all of them here are — you can just do gradient descent on the action sequence:

$$
a_{0:H} \;\leftarrow\; a_{0:H} \;-\; \eta\, \nabla_{a_{0:H}} S(a_{0:H})
$$

Cheaper per unit of improvement, and it uses the Jacobian rather than only zeroth-order information. But: the loss surface through $H$ steps of a learned nonlinear model is riddled with poor local minima, and gradients can vanish or explode along the rollout exactly as in RNN training. DINO-WM implements both and reports **CEM beats GD** empirically. Dreamer sidesteps the issue by amortising into a policy trained with the same gradients but over many rollouts and many states, which acts as an implicit smoother.

Rule of thumb: gradients for *amortised* policies, sampling for *test-time* planning.

### 7.4 Pessimism and uncertainty penalties

To combat model exploitation (§3.10), penalise imagined states the model is unsure about:

$$
\tilde r(\hat s_t, a_t) \;=\; \hat r(\hat s_t, a_t) \;-\; \beta\, u(\hat s_t, a_t)
$$

Estimators for $u$, in increasing order of cost and quality:

- **Ensemble disagreement**: train $M$ dynamics heads, use $u = \operatorname{Var}_m[\hat s^{(m)}_{t+1}]$ or the max pairwise distance.
- **Predicted variance**: if the model outputs a distribution, use its entropy — but note this measures *aleatoric* uncertainty, not the *epistemic* uncertainty you actually want, and conflating them is a common bug.
- **Latent-space density / distance to data**: penalise imagined states far from the replay buffer's latent distribution.

This is the MOPO/MOReL recipe, and it makes the planner conservative in exactly the places where the model is fiction.

### 7.5 The horizon knob

Putting §0.3 and §3.5 together, imagination horizon $H$ trades:

- **Longer $H$** ⇒ better credit assignment for delayed rewards; less dependence on a possibly-bad critic; more compounding model error ($\mathcal{O}(\varepsilon H^2)$).
- **Shorter $H$** ⇒ less model error; more reliance on the critic to summarise everything past the horizon.

Dreamer's $H=15$ with $\lambda$-returns is a principled compromise: $\lambda$ blends across *all* horizons up to $H$ rather than committing to one. Plot task return against $H$ for your own model — the peak location tells you your effective $\varepsilon$, and the shape of the curve is the single most informative diagnostic you can produce.

---

## Part 8 — Evaluating a world model

The recurring mistake in this literature is evaluating world models with generative-model metrics. FID, PSNR, and SSIM measure whether a human would call the video realistic. None of them measures whether the model gets *action consequences* right — and a model can score well on all three while being useless for control (a model that ignores actions and replays a plausible continuation of the video will have excellent FID).

### 8.1 What to measure instead

| Question | Metric | Why |
| --- | --- | --- |
| Does the model track reality over a rollout? | Multi-step latent/state error vs. horizon $h$ | The empirical version of §0.3; the *slope* matters more than the intercept |
| Does the model respond to actions at all? | **Action-conditioning sensitivity**: divergence between rollouts from the same $s_t$ under different $a_t$ | A model that ignores actions can still look great on prediction loss |
| Are counterfactuals correct, not merely different? | Paired interventions with known outcomes | Sensitivity is necessary, not sufficient |
| Does it help decisions? | Real-environment task return vs. random shooting in the true simulator | The only metric that actually matters |
| Is the reward head trustworthy? | Calibration curves; recall on rare success events | Sparse-reward tasks live or die on this |
| Is the representation task-relevant? | Linear probes for object pose, contact state, controllability | Cheap, fast, and diagnostic before you build a planner |
| Is the planner exploiting the model? | Prediction error on **planner-selected** vs. **random** actions | If the gap is large, you are optimising into a blind spot |

🔶 [MiraBench](https://arxiv.org/abs/2605.29360) ("Evaluating Action-Conditioned Reliability in Robotic World Models") is a recent attempt to standardise the second and third rows and is a good paper-presentation candidate.

### 8.2 Model-exploitation tests to run

1. Optimise action sequences **longer** than any seen during training and inspect the resulting states for physical nonsense.
2. Compare prediction error on planner-selected vs. random actions (the gap *is* the exploitation).
3. Add an uncertainty penalty (§7.4) and check whether real-world performance improves — if it does, you were exploiting.
4. Sweep rollout horizon and plot task return; a peak at small $H$ is a model-quality signal.
5. Keep a model-free or true-simulator oracle in the comparison so you know what you are leaving on the table.

### 8.3 A diagnostic worth building once

Plot, on one figure: (a) one-step prediction error, (b) $h$-step prediction error for $h = 1 \ldots 20$, (c) task return under MPC as a function of planning horizon, (d) the same three curves for a model deliberately trained on 10× less data. The relationship between the curves in (a)–(c) — specifically, *where the return curve peaks relative to where the error curve knees* — tells you whether your bottleneck is the model, the planner, or the cost function. Almost no paper reports this and it takes an afternoon.

---

## Part 9 — Running the session

### 9.1 Suggested timing (90 minutes + break)

| Time | Segment | Content |
| --- | --- | --- |
| 0:00–0:10 | Framing | §0.1 master formula, §0.2 POMDP, §1.1 the batter. Put the five-family table on the board and leave it there. |
| 0:10–0:20 | Why | §1.2–1.5. Land the "$N$ imagined rollouts vs. 1 real trial" economics and the strict definition. |
| 0:20–0:35 | Family I | §2.1–2.3 (flow warping, Visual MPC, the blur derivation). §2.4 DIAMOND as the modern answer. |
| 0:35–0:45 | — | **Break** |
| 0:45–1:10 | Family II | §3.1 V-M-C → §3.2 drift → §3.3 RSSM (spend the most time here; the two fixes are the intellectual core of the lecture) → §3.5–3.7 Dreamer lineage → §3.9 DayDreamer. |
| 1:10–1:25 | Families III & IV | §4.1 economics, §4.2 token math, §4.3 where actions live, §4.4–4.6. |
| 1:25–1:40 | Family V | §5.1–5.4 JEPA + collapse + DINO-WM. §5.5 V-JEPA 2.1 if time. |
| 1:40–1:50 | Synthesis | Slide 52 table, §5.7 decoder debate, §6.1 PAN-1 as the null hypothesis. |

If you are short on time, cut §4.2 (token compression) and §3.6 (Dreamer V2 details) first — they are the most self-contained. Do **not** cut §3.2–3.3; the drift → RSSM transition is the argument the whole lecture is built around.

### 9.2 Concept checks

1. Write down the master formula and, for each of the five families, say what $s_t$ is and whether $a_t$ is an input or an output.
2. Why does prediction error compound *quadratically* rather than linearly in the horizon? What does this imply about replan frequency?
3. RSSM splits the state into $h_t$ and $z_t$. Give the specific failure each one fixes, and explain why a single stochastic latent cannot do both jobs.
4. In the RSSM ELBO, what would break if you deleted the KL term? What would break if you deleted the reconstruction term?
5. Derive why an $\ell_2$ frame-prediction loss produces blur. Name three architecturally different fixes and say which family each belongs to.
6. Why does the naive JEPA loss $\|\mathrm{pred}(\mathrm{enc}(o_t),a_t) - \mathrm{enc}(o_{t+1})\|^2$ have a trivial solution? Rank the three anti-collapse strategies by how much you trust them and justify the ranking.
7. Your video tokeniser uses non-causal tubelets. Why can you not use it in a world model?
8. A WAM and an AC-WM are both trained on the same robot data. Name one thing each can do that the other cannot.
9. You have a model with excellent FID and a planner that fails. Give three distinct hypotheses and the experiment that distinguishes them.
10. Under what conditions does PAN-1's "skip the world model" argument fail?

### 9.3 Paper discussion structure

Use the standard template for each of these. Assign one per person.

- **DINO-WM** — *Claim:* pixel prediction is an unnecessary cost; frozen general features + latent dynamics suffice for zero-shot planning. *Evidence to check:* the decoder-loss ablation, and the encoder-swap ablation. *Assumption:* DINOv2 features are a sufficient statistic for the tasks tested. *Failure mode:* tasks whose state is invisible to a frozen encoder (force, mass, internal state). *Next experiment:* what happens on a task requiring a property DINOv2 provably does not encode?
- **Dreamer V3** — *Claim:* fixed hyperparameters across all domains. *Evidence:* Minecraft diamonds. *Assumption:* the robustness transforms (symlog, twohot, return normalisation) are domain-general rather than co-tuned. *Failure mode:* domains where reward scale genuinely carries information.
- **DIAMOND** — *Claim:* visual details matter; discrete compression loses them. *Tension:* with DINO-WM. Resolve it or explain why both are right.
- **DreamZero** — *Claim:* a world action model is a zero-shot policy. *Press on:* what does "zero-shot" mean when the model is fine-tuned on robot data?
- **PAN-1** — *Claim:* you can skip the world model entirely. *Press on:* the "beyond imitation" row of slide 49.
- **μ0** — *Claim:* 3D traces are the right embodiment-agnostic intermediate. *Press on:* is this a principled inductive bias or a reintroduced bottleneck?

### 9.4 Build milestone

Scale to fit the week; the point is the diagnostic curve, not the SOTA number.

**Task.** Learn a one-step dynamics model in a compact state space (start with a PushT or 2D-maze environment where you have ground-truth state; this removes representation quality as a confound).

**Steps.**

1. Fit $\hat s_{t+1} = f_\theta(s_t, a_t)$ on offline random-policy data. Report one-step error.
2. Roll out for $h = 1 \ldots 20$ and **plot prediction error vs. horizon**. Fit the curve — is it linear or quadratic? Compare with the $\mathcal{O}(\varepsilon h^2)$ prediction of §0.3.
3. Plug it into random shooting and CEM (§7.2). Compare task return against random shooting **in the true simulator** — this tells you the cost of your model, isolated from the cost of your planner.
4. **Ablate replan frequency** (execute 1, 2, 5, 10 steps of each plan). This should produce the cleanest curve of the whole exercise and directly demonstrates §0.3.
5. **Ablate the uncertainty penalty** (§7.4) with a 5-member ensemble. Report whether it helps, and measure the prediction-error gap between planner-selected and random actions.

**Stretch goals.** Swap the ground-truth state for a frozen DINOv2 encoding and re-run steps 1–4 (a miniature DINO-WM). Or add a stochastic head and compare deterministic vs. probabilistic prediction on a deliberately multimodal task.

**Deliverable.** The four-panel figure from §8.3 plus two paragraphs: what the horizon curve says about your $\varepsilon$, and whether the useful model was the one with the lowest prediction error.

**Metric that decides the milestone:** *the useful model is the one that improves real decisions, not the one with the sharpest reconstruction.*

### 9.5 Exit ticket

- One result:
- One failure:
- One next action and owner:

---

## Appendix A — Math toolbox

### A.1 The ELBO and the reparameterisation trick

For latent-variable model $p_\theta(o, z) = p_\theta(o\mid z)p(z)$ and variational posterior $q_\phi(z \mid o)$:

$$
\log p_\theta(o) = \log \int p_\theta(o \mid z)p(z)\,dz
= \log \mathbb{E}_{q_\phi}\!\left[\frac{p_\theta(o\mid z) p(z)}{q_\phi(z\mid o)}\right]
\;\ge\; \mathbb{E}_{q_\phi}\!\left[\log \frac{p_\theta(o\mid z)p(z)}{q_\phi(z\mid o)}\right]
$$

by Jensen. Rearranged:

$$
\log p_\theta(o) \;\ge\; \underbrace{\mathbb{E}_{q_\phi}\big[\log p_\theta(o \mid z)\big]}_{\text{reconstruction}} - \underbrace{D_{\mathrm{KL}}\big(q_\phi(z\mid o) \,\|\, p(z)\big)}_{\text{regulariser}} \;=:\; \text{ELBO}
$$

The gap is exactly $D_{\mathrm{KL}}(q_\phi(z\mid o) \,\|\, p_\theta(z \mid o)) \ge 0$ — the ELBO is tight iff the variational posterior equals the true posterior.

**Reparameterisation.** $\nabla_\phi \mathbb{E}_{q_\phi}[f(z)]$ cannot be moved inside the expectation because the distribution depends on $\phi$. For a Gaussian, write $z = \mu_\phi(o) + \sigma_\phi(o)\odot\epsilon$ with $\epsilon\sim\mathcal{N}(0,I)$. Now the randomness is parameter-free:

$$
\nabla_\phi \mathbb{E}_{\epsilon}\big[f(\mu_\phi + \sigma_\phi \odot \epsilon)\big] = \mathbb{E}_{\epsilon}\big[\nabla_\phi f(\mu_\phi + \sigma_\phi\odot\epsilon)\big]
$$

This is *the* enabling trick for Dreamer's backprop-through-dynamics: the whole imagined rollout becomes a deterministic function of $(\theta, \psi, \epsilon_{1:H})$.

**Gaussian KL, closed form** (diagonal, against $\mathcal{N}(0,I)$):

$$
D_{\mathrm{KL}} = \tfrac{1}{2}\sum_{j=1}^{d}\big(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\big)
$$

### A.2 Mixture density networks

$$
p(y \mid x) = \sum_{k=1}^{K}\pi_k(x)\,\mathcal{N}\big(y; \mu_k(x), \operatorname{diag}\sigma^2_k(x)\big)
$$

with $\pi = \operatorname{softmax}(\ell)$ and $\sigma = \exp(\cdot)$ or softplus for positivity. Negative log-likelihood:

$$
\mathcal{L} = -\log \sum_k \exp\Big( \log\pi_k - \tfrac{1}{2}\sum_j\Big[\tfrac{(y_j-\mu_{kj})^2}{\sigma_{kj}^2} + \log(2\pi\sigma_{kj}^2)\Big] \Big)
$$

Implement with `logsumexp` — the naive form underflows immediately.

**Temperature sampling.** $\sigma_k \to \tau\sigma_k$ and $\pi \to \operatorname{softmax}(\log\pi / \tau)$. $\tau>1$ widens both the components and the mixture weights.

### A.3 Straight-through and Gumbel-softmax

**Straight-through** (Dreamer V2/V3). Forward: a hard one-hot sample. Backward: gradient of the soft probabilities.

$$
z \;=\; \operatorname{onehot}(k),\quad k \sim \operatorname{Cat}(\pi)
\qquad\text{implemented as}\qquad
z \;=\; \operatorname{onehot}(k) + \pi - \operatorname{sg}[\pi]
$$

The forward value is $\operatorname{onehot}(k)$ (the $\pi$ terms cancel exactly); the backward gradient is $\partial z/\partial \pi = I$. It is a biased estimator, and it works anyway.

**Gumbel-softmax** is the continuous relaxation: with $g_i \sim \operatorname{Gumbel}(0,1)$,

$$
y_i = \frac{\exp\big((\log\pi_i + g_i)/\tau\big)}{\sum_j \exp\big((\log\pi_j + g_j)/\tau\big)}
$$

which converges to a one-hot categorical sample as $\tau\to 0$. Trades bias for variance against straight-through.

### A.4 KL balancing and free bits

$$
\mathcal{L}_{\mathrm{KL}} = \alpha\, D_{\mathrm{KL}}\big(\operatorname{sg}[q]\,\|\,p\big) + (1-\alpha)\, D_{\mathrm{KL}}\big(q\,\|\,\operatorname{sg}[p]\big), \qquad \alpha = 0.8
$$

The two terms have the *same value* but different gradients: the first updates only $p$ (the prior/dynamics), the second only $q$ (the posterior/encoder). $\alpha = 0.8$ says: mostly make the dynamics model catch up to what the encoder knows, rather than making the encoder dumb enough for the dynamics model.

**Free bits:** $\mathcal{L}_{\mathrm{KL}} \leftarrow \max(\kappa,\ \mathcal{L}_{\mathrm{KL}})$ with $\kappa = 1$ nat. Below the floor there is no gradient, so the model can use its first nat of latent information for free instead of being pushed toward posterior collapse.

### A.5 $\lambda$-returns

$$
V^\lambda_t = \hat r_t + \gamma\Big[(1-\lambda)\,v_\xi(\hat s_{t+1}) + \lambda\,V^\lambda_{t+1}\Big], \qquad V^\lambda_H = v_\xi(\hat s_H)
$$

Expanded, this is the geometric average of all $n$-step returns:

$$
V^\lambda_t = (1-\lambda)\sum_{n=1}^{H-t-1}\lambda^{n-1} G^{(n)}_t \;+\; \lambda^{H-t-1} G^{(H-t)}_t,
\qquad
G^{(n)}_t = \sum_{k=0}^{n-1}\gamma^k \hat r_{t+k} + \gamma^n v_\xi(\hat s_{t+n})
$$

In a *learned* model the interpretation shifts: larger $\lambda$ means more weight on long model rollouts (more model error, less critic bias); smaller $\lambda$ leans on the critic. $\lambda$ is a model-trust parameter.

### A.6 Flow matching and shortcut models

**Flow matching.** Define a probability path from noise $x_0 = \epsilon \sim \mathcal{N}(0,I)$ to data $x_1$ by linear interpolation:

$$
x_\tau = (1-\tau)\epsilon + \tau x_1, \qquad \tau \in [0,1]
$$

The conditional velocity along this path is constant: $\dfrac{dx_\tau}{d\tau} = x_1 - \epsilon$. Regress a network onto it:

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{x_1,\epsilon,\tau}\Big[\big\| v_\theta(x_\tau, \tau, c) - (x_1 - \epsilon)\big\|_2^2\Big]
$$

Sampling integrates $dx/d\tau = v_\theta(x,\tau,c)$ from $\tau=0$ to $1$. The remarkable fact is that regressing the *conditional* velocity (which depends on the specific pair $(\epsilon, x_1)$) yields, at the optimum, the *marginal* velocity field that transports the noise distribution to the data distribution — so a per-sample regression gives you a valid generative ODE.

**Relation to diffusion.** Same family of objects; flow matching with linear paths is a particular (and unusually well-conditioned) choice of noise schedule, and typically needs fewer sampling steps.

**Shortcut models.** Add the step size $d$ as an input, $v_\theta(x, \tau, d)$, and train a self-consistency condition so that one big step matches two small ones:

$$
v_\theta(x, \tau, 2d) \;\approx\; \tfrac{1}{2}\Big[ v_\theta(x,\tau,d) + v_\theta\big(x + d\,v_\theta(x,\tau,d),\; \tau+d,\; d\big)\Big]
$$

This trains few-step (even one-step) generation *directly*, rather than distilling a many-step model afterwards — which is what makes flow-based world models fast enough to roll out inside an RL loop (Dreamer V4).

### A.7 The simulation lemma

**Setting.** MDPs $M = (\mathcal{S},\mathcal{A},p,r,\gamma)$ and $\hat M = (\mathcal{S},\mathcal{A},\hat p,r,\gamma)$ sharing rewards, $r \in [0,R_{\max}]$, with $\max_{s,a} D_{\mathrm{TV}}(p(\cdot|s,a), \hat p(\cdot|s,a)) \le \varepsilon$.

**Step 1 — distributions.** Let $d_t, \hat d_t$ be the state distributions under a shared policy $\pi$. The hybrid argument (§0.3) gives $D_{\mathrm{TV}}(d_t,\hat d_t) \le t\varepsilon$.

**Step 2 — finite horizon.** With $|\mathbb{E}_p f - \mathbb{E}_q f| \le 2\|f\|_\infty D_{\mathrm{TV}}(p,q)$:

$$
|J_H - \hat J_H| \le \sum_{t=0}^{H-1} 2R_{\max}\,t\,\varepsilon = R_{\max}\varepsilon\, H(H-1)
$$

**Step 3 — discounted.** Summing $\gamma^t \cdot 2R_{\max} t \varepsilon$ over $t \ge 0$ and using $\sum_t t\gamma^t = \gamma/(1-\gamma)^2$:

$$
|V^\pi_M - V^\pi_{\hat M}| \;\le\; \frac{2\gamma\,\varepsilon\,R_{\max}}{(1-\gamma)^2}
$$

**Interpretation.** Two factors of $1/(1-\gamma)$: one because errors accumulate along the trajectory, one because each wrong state is then evaluated for the remaining horizon. This is why model-based RL is hard in a way that is not fixed by a bigger network.

### A.8 Total variation, and why it is the right distance here

$$
D_{\mathrm{TV}}(p,q) = \sup_{A} |p(A) - q(A)| = \tfrac{1}{2}\|p - q\|_1
$$

TV is the natural distance for the simulation lemma because it directly bounds the difference in expectation of any bounded function — including any reward function and any planner cost. KL is *not* directly usable here (it is not a metric and can be infinite for distributions with different support), but Pinsker's inequality relates them:

$$
D_{\mathrm{TV}}(p,q) \le \sqrt{\tfrac{1}{2} D_{\mathrm{KL}}(p \| q)}
$$

which is how a likelihood-trained model (which bounds KL) gives you a return guarantee (which needs TV) — and note the square root, which means halving your KL loss buys you only a $\sqrt{2}$ improvement in the return bound. This is the quantitative form of §0.5's objective-mismatch problem.

---

## Appendix B — Notation

| Symbol | Meaning |
| --- | --- |
| $o_t$ | observation (image, sensor reading) at time $t$ |
| $s_t$ | state — whatever the family uses as its state representation |
| $z_t$ | stochastic latent / learned code |
| $h_t$ | deterministic recurrent state; more generally, memory of the past |
| $a_t$ | action |
| $r_t$, $\hat r_t$ | real and predicted reward |
| $g$ | goal (language instruction, goal image, or goal latent) |
| $q_t$ | proprioception |
| $H$ | rollout / imagination / planning horizon |
| $\gamma$, $\lambda$ | discount factor; $\lambda$-return mixing parameter |
| $\pi_\psi$, $v_\xi$ | actor and critic |
| $p$, $q$ | model prior and variational posterior |
| $\operatorname{sg}[\cdot]$ | stop-gradient |
| $\varepsilon$ | one-step model error in total variation |
| $\sigma$, $\tau$ | diffusion noise level; flow-matching time (also MDN temperature) |
| $\mathrm{enc}$, $\mathrm{dec}$, $\mathrm{pred}$ | encoder, decoder, latent predictor |
| **AC-WM** | action-conditioned world model |
| **WAM** / **VAM** | world action model / video action model |
| **IDM** | inverse dynamics model — infers $a_t$ from $(o_t, o_{t+1})$ |
| **RSSM** | recurrent state-space model |
| **JEPA** | joint-embedding predictive architecture |
| **MPC** / **CEM** / **MPPI** | model-predictive control; cross-entropy method; model-predictive path integral |

---

## Appendix C — Reading list

**On the slides**

- Finn, Goodfellow, Levine (2016). *Unsupervised Learning for Physical Interaction through Video Prediction.* [arXiv:1605.07157](https://arxiv.org/abs/1605.07157)
- Ebert, Finn et al. (2018). *Visual Foresight: Model-Based Deep RL for Vision-Based Robotic Control.* [arXiv:1812.00568](https://arxiv.org/abs/1812.00568)
- Ha & Schmidhuber (2018). *World Models.* [arXiv:1803.10122](https://arxiv.org/abs/1803.10122) · [worldmodels.github.io](https://worldmodels.github.io/)
- Hafner et al. (2019). *Learning Latent Dynamics for Planning from Pixels* (PlaNet). [arXiv:1811.04551](https://arxiv.org/abs/1811.04551)
- Hafner et al. (2020). *Dream to Control: Learning Behaviors by Latent Imagination* (Dreamer V1). [arXiv:1912.01603](https://arxiv.org/abs/1912.01603)
- Hafner et al. (2021). *Mastering Atari with Discrete World Models* (Dreamer V2). [arXiv:2010.02193](https://arxiv.org/abs/2010.02193)
- Hafner et al. (2023). *Mastering Diverse Domains through World Models* (Dreamer V3). [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
- Hafner et al. (2025). *Training Agents Inside of Scalable World Models* (Dreamer V4).
- Wu\*, Escontrela\*, Hafner\* et al. (2022). *DayDreamer: World Models for Physical Robot Learning.* [arXiv:2206.14176](https://arxiv.org/abs/2206.14176)
- Yan et al. (2025). *ElasticTok.* ICLR 2025.
- Pai\*, Achenbach\*, Montesinos, Forrai, Mees\*, Nava\* (2025). *mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs.*
- Ye et al. (2026). *World Action Models are Zero-shot Policies* (DreamZero). [DreamZero.pdf](https://dreamzero0.github.io/DreamZero.pdf)

**Requested additions**

- Zhou, Pan, LeCun, Pinto. *DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning.* [arXiv:2411.04983](https://arxiv.org/abs/2411.04983)
- Alonso, Jelley, Micheli, Kanervisto, Storkey, Pearce, Fleuret (2024). *Diffusion for World Modeling: Visual Details Matter in Atari* (DIAMOND). [arXiv:2405.12399](https://arxiv.org/abs/2405.12399) · [diamond-wm.github.io](https://diamond-wm.github.io/)
- Mur-Labadia et al. *V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning.* [arXiv:2603.14482](https://arxiv.org/abs/2603.14482)
- *MIRA: Multiplayer Interactive World Models with Representation Autoencoders.* [mira-wm.com](https://mira-wm.com/) · [github.com/mira-wm/mira](https://github.com/mira-wm/mira) · [arXiv:2607.05352](https://arxiv.org/abs/2607.05352)
- *μ0: 3D Interaction-Trace World Model.* [mu0-wm.github.io](https://mu0-wm.github.io/)
- Pantograph. *PAN-1.* [pantograph.com/journal/pan-1](https://pantograph.com/journal/pan-1)

**Useful background not cited in the deck**

- Karras, Aittala, Aila, Laine (2022). *Elucidating the Design Space of Diffusion-Based Generative Models* (the EDM parameterisation used in §2.4). [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)
- Lipman et al. (2023). *Flow Matching for Generative Modeling.* [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- Micheli, Alonso, Fleuret (2023). *Transformers are Sample-Efficient World Models* (IRIS). [arXiv:2209.00588](https://arxiv.org/abs/2209.00588)
- Hansen, Su, Wang (2024). *TD-MPC2: Scalable, Robust World Models for Continuous Control.* [arXiv:2310.16828](https://arxiv.org/abs/2310.16828)
- Janner, Fu, Zhang, Levine (2019). *When to Trust Your Model: Model-Based Policy Optimization* (MBPO — the compounding-error analysis). [arXiv:1906.08253](https://arxiv.org/abs/1906.08253)
- Yu et al. (2020). *MOPO: Model-based Offline Policy Optimization* (uncertainty penalties). [arXiv:2005.13239](https://arxiv.org/abs/2005.13239)
- Bardes et al. (2022). *VICReg: Variance-Invariance-Covariance Regularization.* [arXiv:2105.04906](https://arxiv.org/abs/2105.04906)
- *MiraBench: Evaluating Action-Conditioned Reliability in Robotic World Models.* [arXiv:2605.29360](https://arxiv.org/abs/2605.29360)

---

## Appendix D — Slide-to-section map

| Slides | Content | Section |
| --- | --- | --- |
| 1–9 | Course admin, mid-term feedback, projects | — (see [week-08-world-models.md](week-08-world-models.md)) |
| 10 | Batter / subconscious world model | [§1.1](#11-the-batter-slide-10) |
| 11 | World models vs policies | [§1.2](#12-policies-versus-world-models-slide-11) |
| 12 | Why world models — $N$ rollouts, 1 trial | [§1.4](#14-what-prediction-buys-you-slide-12) |
| 13 | Data-driven simulators | [§1.3](#13-a-world-model-is-a-data-driven-simulator-slide-13) |
| 14 | Strict definition | [§1.5](#15-the-strict-definition-slide-14) |
| 15 | Pixel AC-WMs — Finn 2016 | [§2.1](#21-finn-goodfellow--levine-2016-predict-motion-not-pixels-slide-15) |
| 16–17 | Visual Foresight / Visual MPC | [§2.2](#22-visual-foresight--visual-mpc-ebert-finn-et-al-2018-slides-1617) |
| — | 🔶 DIAMOND, MIRA | [§2.4](#24--diamond-pixel-world-models-done-properly), [§2.5](#25--mira-interactive-pixel-world-models-at-scale) |
| 18–19 | Latent AC-WMs — general structure | [Part 3 intro](#part-3--family-ii-latent-action-conditioned-world-models) |
| 20–25 | Ha & Schmidhuber V-M-C | [§3.1](#31-the-original-ha--schmidhuber-2018-vmc-slides-2025) |
| 26 | Why imagination drifts | [§3.2](#32-why-imagination-without-a-prior-drifts-slide-26) |
| 27 | RSSM training | [§3.3](#33-rssm-and-planet-split-the-state-slide-27) |
| 28 | RSSM closed-loop imagination | [§3.4](#34-rssm-closed-loop-inference-and-imagination-slide-28) |
| 29 | Dreamer V1 vs PlaNet | [§3.5](#35-dreamer-v1-amortise-the-planning-slide-29) |
| 30 | DayDreamer | [§3.9](#39-daydreamer-does-it-survive-contact-with-hardware-slide-30) |
| 31 | The Matrix Dojo | [§3.10](#310-the-matrix-dojo-slide-31) |
| 32 | Dreamer lineage | [§3.6](#36-dreamer-v2-categorical-latents-and-kl-balancing-slide-32)–[§3.8](#38-dreamer-v4-decoupling-video-from-actions-slide-33) |
| 33 | Dreamer V4 | [§3.8](#38-dreamer-v4-decoupling-video-from-actions-slide-33) |
| 34–36 | Video tokenisation and compression | [§4.2](#42-the-token-economics-problem-slides-3536) |
| 37 | Where do actions live? | [§4.3](#43-where-do-actions-live-slide-37) |
| 38–42 | VLA sample efficiency → video backbones | [§4.1](#41-the-economic-argument-vlas-are-sample-inefficient-slides-3941) |
| 43–46 | mimic-video | [§4.4](#44-mimic-video-the-vam-instantiation-slides-4346) |
| 47–48 | DreamZero | [§4.5](#45-dreamzero-the-wam-instantiation-slides-4748) |
| 49 | WAMs vs AC-WMs | [§4.6](#46-wams-versus-ac-wms-slide-49) |
| 50–51 | JEPA | [§5.1](#51-the-jepa-objective-slide-51)–[§5.3](#53-three-anti-collapse-strategies-slide-51) |
| — | 🔶 DINO-WM, V-JEPA 2.1, μ0, PAN-1 | [§5.4](#54--dino-wm-the-frozen-encoder-instantiation)–[§6.1](#61--pantograph-pan-1-goal-conditioned-rl-from-action-free-video) |
| 52 | Conclusion — five families, open questions | [§0.1](#01-the-master-formula), [§5.7](#57-does-the-decoder-ever-earn-its-keep) |
| 53–54 | Thanks, references | [Appendix C](#appendix-c--reading-list) |

---

[← Sequence Modeling](week-07-sequence-modeling.md) · [Slide gallery](week-08-world-models.md) · [Next: Generalist Policies →](week-09-generalist-policies.md)
