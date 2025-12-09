# Chapter 10: Reinforcement Learning from Human Feedback and Value Alignment


## Motivation and Challenges in Value Alignment for Language Models

Designing a reward function that captures exactly what we want from a language model is extremely difficult. In open-ended tasks like dialogue or summarization, we cannot easily hand-craft a numeric reward for “good” behavior. This is where Reinforcement Learning from Human Feedback (RLHF) comes in. RLHF is a strategy to achieve value alignment – ensuring an AI’s behavior aligns with human preferences and values – by using human feedback as the source of reward. Instead of explicitly writing a reward function, we ask humans to compare or rank outputs, and use those preferences as a training signal. Humans find it much easier to choose which of two responses is better than to define a precise numerical reward for each outcome. For example, it's simpler for a person to say which of two summaries is more accurate and polite than to assign an absolute “score” to a single summary. By leveraging these relative judgments, RLHF turns human preference data into a reward model that guides the training of our policy (the language model) toward preferred behaviors.

However, using human feedback poses its own challenges. Human preferences are noisy and subjective, and the process of training with them can be unstable if not done carefully. Early large-scale successes (such as OpenAI’s InstructGPT) demonstrated that RLHF can significantly improve alignment with user intentions, but also highlighted the complexity of the process. In a typical RLHF setup, there are multiple phases of training (supervised and RL), and careful tuning is needed to prevent the model from gaming the learned reward or drifting from its original fluent writing style. In short, RLHF is powerful but complex, motivating research into simpler alternatives that still leverage human preferences.

## The RLHF Training Pipeline

![Alt text](images/llm_rlhf.png)

To train a language model with human feedback, practitioners usually follow a three-stage pipeline. Each stage uses a different training paradigm (supervised learning or reinforcement learning) to gradually align the model with what humans prefer:

1. Supervised Fine-Tuning (SFT) – Start with a pretrained model and fine-tune it on demonstrations of the desired behavior. For example, using a dataset of high-quality question-answer pairs or summaries written by humans, we train the model to imitate these responses. This teacher forcing stage grounds the model in roughly the right style and tone (as discussed in earlier chapters on imitation learning). By the end of SFT, the model (often called the reference model) is a strong starting point that produces decent responses, but it may not perfectly adhere to all subtle preferences or values because it was only trained to imitate the data.

2. Reward Model Training from Human Preferences – Next, we collect human feedback in the form of pairwise preference comparisons. For many prompts, humans are shown two model-generated responses and asked which one is better (or if they are equally good). From these comparisons, we learn a reward function $r_\phi(x,y)$ (parameterized by $\phi$) that predicts which response is more preferable for a given input x. A simple and commonly used approach is the Bradley–Terry model, which assumes the probability that a human prefers response $y_i$ over $y_j$ is given by a logistic function of their underlying rewards:

    $$P(y_i \succ y_j \mid x) =
    \frac{\exp\!\left(r_\phi(x, y_i)\right)}
    {\exp\!\left(r_\phi(x, y_i)\right) + \exp\!\left(r_\phi(x, y_j)\right)}$$

    In other words, the higher the reward $r_\phi$ for an outcome, the more likely that outcome is to be chosen as the better one. We fit the reward model $\phi$ by maximizing the likelihood of the human comparison data (equivalently, minimizing a cross-entropy loss). If humans indicated that $y_i$ was preferable to $y_j$, the training objective will push $r_\phi(x,y_i)$ to be larger than $r_\phi(x,y_j)$, in line with the logistic formula. This process yields a learned reward function that aligns with human judgments of quality, at least up to some constant scaling or shifting (note that adding a constant to $r_\phi$ doesn’t change the preference ordering). The reward model effectively becomes a stand-in for "human values": given a new prompt and a candidate answer, it provides a scalar score for how desirable that answer is according to human preferences.

3. Reinforcement Learning Fine-Tuning – In the final stage, we use the learned reward model as a surrogate reward signal to fine-tune the policy (the language model) via reinforcement learning. The policy $\pi_\theta(y|x)$ (with parameters $\theta$) is updated to maximize the expected reward $r_\phi(x,y)$ of its outputs, while also staying close to the behavior of the reference model from stage 1. This last point is crucial: if we purely maximize the reward model’s score, the policy might exploit flaws in $r_\phi$ (a form of “reward hacking”) or produce unnatural outputs that, for example, repeat certain high-reward phrases. To prevent the policy from straying too far, RLHF algorithms introduce a Kullback–Leibler (KL) penalty that keeps the new policy $\pi_\theta$ close to the reference policy $\pi_{\text{ref}}$ (often the SFT model). In summary, the RL objective can be written as:

    $$J(\pi_\theta) =
    \mathbb{E}_{x \sim D,\; y \sim \pi_\theta(\cdot \mid x)}\left[ r_\phi(x, y) \right] - \beta \, \mathbb{E}_{x}
    \left[ D_{\mathrm{KL}}\!\left( \pi_\theta(\cdot \mid x)\,\|\,\pi_{\text{ref}}(\cdot \mid x) \right) \right]$$

    where $\beta>0$ controls the strength of the penalty. Intuitively, this objective asks the new policy to generate high-reward answers on the training prompts, but it subtracts points if $\pi_\theta$ deviates too much from the original model’s distribution (as measured by KL divergence). The KL term thus acts as a regularizer encouraging conservatism: the policy should only change as needed to gain reward, and not forget its broadly learned language skills or go out-of-distribution. In practice, this RL optimization is performed using Proximal Policy Optimization (PPO) (introduced in Chapter 7) or a similar policy gradient method. PPO is well-suited here because it naturally limits the size of each policy update (via the clipping mechanism), complementing the KL penalty to maintain stability.

    >   Why KL Penalty? The KL penalty can be seen as implementing a form of trust-region or regularized RL. It is often directly added into the reward function for optimization. For example, the OpenAI InstructGPT implementation defined an augmented reward $r'(x,y) = r_\phi(x,y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$. Maximizing this $r'$ via RL (with $\beta$ appropriately tuned) has the effect of penalizing changes that make $\pi_\theta$ very confident on sequences that the reference model found unlikely. This keeps the language model’s style and content from drifting in undesirable ways (such as becoming too terse, too verbose, or incoherent) while it maximizes the learned reward.

Through this pipeline – SFT, reward modeling, and RL fine-tuning – we obtain a policy that hopefully excels at the task as defined implicitly by human preferences. Indeed, RLHF has enabled large language models to better follow instructions, avoid blatantly harmful content, and generally be more helpful and aligned with user expectations than they would be out-of-the-box. That said, the full RLHF procedure involves training multiple models (a reward model and the policy) and carefully tuning hyperparameters (like $\beta$ and PPO clip thresholds). The process can be unstable; for instance, if $\beta$ is too low, the policy might mode-collapse to only a narrow set of high-reward answers, whereas if $\beta$ is too high, the policy might hardly improve at all. Researchers have described RLHF as a “complex and often unstable procedure” that requires balancing between reward optimization and avoiding model drift. This complexity has spurred interest in whether we can achieve similar alignment benefits without a full reinforcement learning loop. 

## Direct Preference Optimization: RLHF without RL?

Direct Preference Optimization (DPO) is a recently introduced alternative to the standard RLHF fine-tuning stage. The key idea of DPO is to solve the RLHF objective in closed-form, and then optimize that solution directly via supervised learning. DPO manages to sidestep the need for sampling-based RL (like PPO) by leveraging the mathematical structure of the RLHF objective we defined above.

Recall that in the RLHF setting, our goal is to find a policy $\pi^*(y|x)$ that maximizes reward while staying close to a reference policy. Conceptually, we can write the optimal policy for a given reward function in a Boltzmann (exponential) form. In fact, it can be shown (see e.g. prior work on KL-regularized RL) that the optimizer of $J(\pi)$ occurs when $\pi$ is proportional to the reference policy times an exponential of the reward:

$$\pi^*(y \mid x) \propto \pi_{\text{ref}}(y \mid x)\,
\exp\!\left(\frac{1}{\beta}\, r_\phi(x, y)\right)$$

This equation gives a closed-form solution for the optimal policy in terms of the reward function $r_\phi$. It makes sense: actions $y$ that have higher human-derived reward should be taken with higher probability, but we temper this by $\beta$ and weight by the reference probabilities $\pi_{\text{ref}}(y|x)$ so that we don’t stray too far. If we were to normalize the right-hand side, we’d write:

$$\pi^*(y \mid x)
=
\frac{
\pi_{\text{ref}}(y \mid x)\,
\exp\!\left(\frac{r_\phi(x,y)}{\beta}\right)
}{
\sum_{y'} \pi_{\text{ref}}(y' \mid x)\,
\exp\!\left(\frac{r_\phi(x,y')}{\beta}\right)
}$$

Here the denominator is a partition function $Z(x)$ summing over all possible responses $y'$ for input $x$. This normalization involves a sum over the entire response space, which is astronomically large for language models – hence we cannot directly compute $\pi^*(y|x)$ in practice. This intractable sum is exactly why the original RLHF approach uses sampling-based optimization (PPO updates) to approximate the effect of this solution without computing it explicitly.

DPO’s insight is that although we cannot evaluate the normalizing constant $Z(x)$ easily, we can still work with relative probabilities. In particular, for any two candidate responses $y_+$ (preferred) and $y_-$ (dispreferred) for the same context $x$, the normalization $Z(x)$ cancels out if we look at the ratio of the optimal policy probabilities. Using the form above:


$$\frac{\pi^*(y^+ \mid x)}{\pi^\ast(y^- \mid x)}
=
\frac{\pi_{\text{ref}}(y^+ \mid x)\,
\exp\!\left(\frac{r_\phi(x, y^+)}{\beta}\right)}
{\pi_{\text{ref}}(y^- \mid x)\,
\exp\!\left(\frac{r_\phi(x, y^-)}{\beta}\right)}
=
\frac{\pi_{\text{ref}}(y^+ \mid x)}
{\pi_{\text{ref}}(y^- \mid x)}
\exp\!\left(
\frac{1}{\beta}
\big[
r_\phi(x, y^+) - r_\phi(x, y^-)
\big]
\right)$$

Taking the log of both sides, we get a neat relationship:

$$\frac{1}{\beta}
\big( r_\phi(x, y^{+}) - r_\phi(x, y^{-}) \big)
=
\big[ \log \pi^\ast(y^{+} \mid x) - \log \pi^\ast(y^{-} \mid x) \big]
-
\big[ \log \pi_{\text{ref}}(y^{+} \mid x) - \log \pi_{\text{ref}}(y^{-} \mid x) \big]$$

The term in brackets on the right is the difference in log-probabilities that the optimal policy $\pi^*$ assigns to the two responses (which in turn would equal the difference in our learned policy’s log-probabilities if we can achieve optimality). What this equation tells us is: the difference in reward between a preferred and a rejected response equals the difference in log odds under the optimal policy (minus a known term from the reference model). In other words, if $y_+$ is better than $y_-$ by some amount of reward, then the optimal policy should tilt its probabilities in favor of $y_+$ by a corresponding factor.

Crucially, the troublesome normalization $Z(x)$ is gone in this ratio. We can rearrange this relationship to directly solve for policy probabilities in terms of rewards, or vice-versa. DPO leverages this to cut out the middleman (explicit RL). Instead of updating the policy via trial-and-error with PPO, DPO directly adjusts $\pi_\theta$ to satisfy these pairwise preference constraints. Specifically, DPO treats the problem as a binary classification: given a context $x$ and two candidate outputs $y_+$ (human-preferred) and $y_-$ (human-dispreferred), we want the model to assign a higher probability to $y_+$ than to $y_-$, with a confidence that grows with the margin of preference. We can achieve this by maximizing the log-likelihood of the human preferences under a sigmoid model of the log-probability difference.

In practice, the DPO loss for a pair $(x, y_+, y_-)$ is something like:

$$\ell_{\text{DPO}}(\theta)
= - \log \sigma \!\left(
\beta\,
\big[ \log \pi_\theta(y^{+} \mid x) - \log \pi_\theta(y^{-} \mid x) \big]
\right)$$

where $\sigma$ is the sigmoid function. This loss is low (i.e. good) when $\log \pi_\theta(y_+|x) \gg \log \pi_\theta(y_-|x)$, meaning the model assigns much higher probability to the preferred outcome – which is what we want. If the model hasn’t yet learned the preference, the loss will be higher, and gradient descent on this loss will push $\pi_\theta$ to increase the probability of $y_+$ and decrease that of $y_-$. Notice that this is very analogous to the Bradley-Terry formulation earlier, except now we embed the reward model inside the policy’s logits: effectively, $\log \pi_\theta(y|x)$ plays the role of a reward score for how good $y$ is, up to the scaling factor $1/\beta$. In fact, the DPO derivation can be seen as combining the preference loss on $r_\phi$ with the $\pi^*$ solution formula to produce a preference loss on $\pi_\theta$. The original DPO paper calls this approach “your language model is secretly a reward model” – by training the language model with this loss, we are directly teaching it to act as if it were the reward model trying to distinguish preferred vs. non-preferred outputs.

> Derivation Summary: We started with the RLHF objective and saw that the optimal policy would take the form $\pi^*(y)\propto \pi_{\text{ref}}(y)\exp(r(y)/\beta)$. While we couldn’t compute $\pi^*$ explicitly (due to the sum inside $Z(x)$), we derived a condition for any pair of outputs that should hold at optimum: the reward difference matches the log-probability difference. DPO trains the model to satisfy that condition for the human-labeled preference pairs by treating it as a logistic regression problem. This avoids sampling a trajectory and computing cumulative rewards – we directly use the static dataset of comparisons.

Advantages of DPO: DPO offers several practical benefits over the traditional RLHF fine-tuning approach:

1. Simplicity: DPO reduces the alignment problem to simple supervised learning – optimizing a classification loss – rather than a full reinforcement learning loop. This means we don’t need to worry about training stability issues from PPO (no more juggling advantage estimation, value functions, or delicate learning rate schedules for RL). We also don’t need to continuously sample from the model during training; we can use a fixed set of human comparisons. In essence, it eliminates the reinforcement learning step entirely while achieving a similar effect.

2. No Separate Reward Model Needed: In standard RLHF, one trains a reward model $r_\phi$ and then uses it in the loop with PPO. DPO collapses this two-step process into one – the preference data directly trains the policy. In fact, DPO can be interpreted as implicitly learning a reward function inside the model’s parameters. An immediate benefit is that we avoid the cost of training a separate reward network and maintaining it. (In implementations, one often still keeps a frozen copy of the reference model to compute the $\log \pi_{\text{ref}}$ term, but that’s just the original SFT model, not a learned network.)

3. Efficiency and Stability: Experience so far suggests that DPO is more sample-efficient and stable. For example, one report noted that DPO required significantly less preference data to reach comparable performance to a PPO-based approach. This may be because DPO directly uses each human-labeled comparison in a gradient update, whereas PPO might need many gradient steps and samples to gradually nudge the policy in the right direction. Additionally, DPO has fewer moving parts to tune. The main hyper-parameter is the scale factor $\beta$, which controls how strictly the model is pushed to distinguish preferred vs. rejected outputs. In contrast, RLHF with PPO has $\beta$ (or an equivalent KL weight), plus all the PPO-specific parameters (clip threshold, value loss weight, etc.). DPO’s developers found it to be “stable, performant, and computationally lightweight” in aligning language models, which is a big win for practical deployment.

4. Competitive Performance: Perhaps most importantly, DPO achieves results on par with or better than the classic RLHF. In the original DPO paper, fine-tuning a language model with DPO matched or exceeded PPO-based RLHF on several tasks. For instance, DPO-trained models were better at controlling the sentiment of generated text (when asked to produce positive or negative outputs) than models tuned with PPO. In tasks like summarization and single-turn dialogue, DPO models achieved equivalent or slightly improved quality compared to PPO-trained models, all while being simpler to train. This is a promising result: it suggests we don’t pay a performance price for ditching the RL step – we might even gain some quality or consistency improvements.

Of course, DPO is a relatively new method and not a silver bullet. It makes certain assumptions (for example, it assumes the form of the optimal policy and uses the reference model probabilities in the loss) and it optimizes the model on the given preference data without explicit consideration of long-horizon effects (though in language modeling each prompt–response is usually treated as an episode anyway). Nonetheless, the emergence of DPO is an exciting development in the quest for better alignment techniques.

## Applications and Recent Developments

Both RLHF and DPO have been employed in training large-scale models, and understanding their use in practice helps appreciate their impact on current AI systems. A notable example is the open-source model Mistral-7B (2023), which underwent an instruction-tuning process that included human preference optimization. The training pipeline for Mistral combined supervised instruction fine-tuning with a DPO-style preference optimization, enabling a small (7-billion-parameter) model to outperform some larger predecessors on helpfulness benchmarks. Another high-profile case is Meta AI’s LLaMa 3 models (2024). The LLaMa-3 series introduced an improved alignment training recipe, reportedly leveraging both PPO-based RLHF and DPO in an iterative fashion. For instance, one public example showed fine-tuning a LLaMa-3 8B model using SageMaker’s DPOTrainer: human annotators first created a dataset of ranked responses, and then the model was directly preference-tuned via DPO on those comparisons. The result was a model that better adhered to user instructions and organizational values, achieved with a simpler training method than traditional RLHF.

These real-world applications underscore that DPO is not just a theoretical nicety – it scales to large models and delivers robust performance. In fact, the success of DPO has sparked a flurry of research into related methods and refinements. For example, some works explore variants like Implicit Preference Optimization (IPO) or techniques to handle biases (e.g. avoiding a tendency to always prefer longer answers). Nonetheless, the core principle remains: use human preference data in the most straightforward way possible to adjust the model’s policy.

In summary, Reinforcement Learning from Human Feedback has become a cornerstone of aligning large language models with what humans want. It tackles the fundamental problem of value alignment – bridging the gap between what we truly care about in a task and what we can explicitly encode as a reward. RLHF’s three-step pipeline (supervised tuning, reward modeling, RL fine-tuning) has proven effective at producing helpful and harmless AI assistants, but it introduces complexity. Direct Preference Optimization offers a fresh take by collapsing the problem into a single, stable optimization that more directly connects human preferences to model behavior. As we have seen, both approaches rely on the notion of using human judgments (pairwise comparisons via the Bradley-Terry model, in our discussion) as a source of truth for what the reward should be. By training models from this feedback, we imbue them with a representation of human values and preferences that is otherwise nearly impossible to hand-specify. Going forward, these techniques – and refinements thereof – will likely remain central in training AI systems that are not only intelligent, but aligned with human objectives. The ongoing evolution from RLHF to DPO and beyond represents progress toward simpler, more efficient value alignment, allowing us to better steer AI behavior with humans at the helm.