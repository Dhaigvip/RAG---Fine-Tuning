# ML Fundamentals — Resource Selection (prerequisite for fine-tuning)

## Problem statement
The fine-tuning track (Sept 17–20) assumes vocabulary and intuition that nothing earlier in this plan actually teaches: what a loss function is, what gradient descent is doing to the weights, why a model overfits and what regularization does about it, why data gets split into train/validation/test, and the bias-variance tradeoff those hyperparameters are balancing. Without this, Sept 17–20 risks becoming "follow the LoRA tutorial steps" rather than "understand what LoRA is doing to the weights and why rank, alpha, and learning rate matter" — the actual stated goal for the fine-tuning track ("learn the end-to-end mechanics, not just calling an API").

The constraint on the fix: this project is explicitly *not* going deep into ML as its own subject — it's a RAG/fine-tuning project, not an ML fundamentals course. So the fix needs to add just enough grounding for the fine-tuning days to be genuinely understood, in the smallest amount of time that achieves that, not a comprehensive ML education.

## Solution
A targeted, video-based resource stack (~3–4 hours total) covering only the specific concepts Sept 17–20 leans on, instead of a structured multi-week course that would cover far more than needed (most structured courses also re-teach things already covered by the RAG build — embeddings, basic LLM concepts — which is wasted time here).

## Landscape considered

**Andrew Ng's Machine Learning Specialization (Coursera)** — ~4 months, full mathematical rigor (derives cost functions, gradient descent by hand), gold-standard for someone becoming an ML practitioner. **Rejected**: far exceeds "not going deep" — both in depth and in time budget.

**Google Machine Learning Crash Course** (2024 rebuild) — ~15 hours, free, interactive notebooks and video, broad survey from classical ML through LLMs and AutoML. **Rejected as the primary resource**: good material, but broad rather than targeted — a large fraction (embeddings, LLM basics) duplicates what the RAG build already taught, and 15 hours doesn't fit a half-day slot.

**fast.ai — Practical Deep Learning for Coders** — ~7 weeks, code-first ("build a working model in week one, learn theory after"). **Rejected**: excellent for someone building deep learning models from scratch, but scoped well beyond "enough understanding to follow along" for a project where the actual model-building is LoRA fine-tuning on an existing model, not architecture design.

**StatQuest (Josh Starmer, YouTube)** — **chosen (primary).** Free, modular 5–15 minute videos, each covering exactly one concept (gradient descent, loss functions, regularization, bias-variance tradeoff, train/validation/test split) with visual intuition and no derivations. Modularity is the key advantage here: pick only the videos that map to Sept 17–20's actual content instead of sitting through a course structured around someone else's curriculum.

**3Blue1Brown — "Neural Networks" series** — **chosen (primary).** Free, 4 videos, ~1 hour total. Visual, mechanism-first explanation of what a neural network is and how backpropagation actually works — directly relevant since fine-tuning means adjusting a neural network's weights, and this is widely regarded as the clearest short explanation of that mechanism available anywhere.

**Andrej Karpathy — "Neural Networks: Zero to Hero" (micrograd)** — free, hands-on: build a backpropagation engine from scratch in code. **Deferred, not rejected**: matches the project's mechanism-based, hands-on learning style well, but is genuinely deeper (~2–3 hours for just the first video) than "enough understanding" requires. Worth returning to later, purely out of interest, once the fine-tuning track is done — not a blocker for Sept 17.

## Chosen resource stack (~3–4 hours)
1. **3Blue1Brown, Neural Networks series** (all 4 videos, ~1 hr) — what a network is, how it computes, how backpropagation adjusts weights.
2. **StatQuest, curated videos** (~2 hrs) — gradient descent (the core algorithm, watch both the "step-by-step" and "clearly explained" ones), loss functions (cross-entropy for classification since that's what LLM fine-tuning uses), overfitting & regularization (L1/L2, and specifically the concept behind why LoRA constrains the update), bias-variance tradeoff, train/validation/test split.
3. Skip: linear/logistic regression from-scratch derivations, decision trees, SVMs, clustering, and other classical-ML topics StatQuest also covers — not load-bearing for fine-tuning an existing LLM with LoRA.

## What's deferred
- Karpathy's micrograd build (see above) — genuinely valuable, not required for this project's scope.
- Any classical ML algorithm coverage beyond what's needed to understand gradient descent/loss/regularization (decision trees, SVMs, clustering, etc.) — this project never trains a classical ML model, so there's no reason to learn algorithms it won't use.

## Schedule placement (decided Sept 14)
Inserted as a new half-day slot between the RAG track (ends Sept 15–16) and the fine-tuning track (Sept 17–20), rather than folded into Sept 17 or left as unscheduled prerequisite viewing — keeps Sept 17's fine-tuning-fundamentals content from being crammed alongside brand-new ML vocabulary in the same session. See `learning-plan-sept12-20.md` for the updated day-by-day.

**Started early (Sept 15)**: the RAG track wrapped ahead of the original day-by-day (generation + the FastAPI endpoint both done by Sept 16), so this slot is being run now rather than waiting for Sept 17 — the plan's dates are a pace guide, not a hard schedule, per how this project has already flexed once before (the retrieval-eval build ran ahead of its original day too).

## Watch list — confirmed links (added Sept 15)
The "Chosen resource stack" above named the concepts; this is the actual ordered, clickable list, verified against StatQuest's own maintained video index (statquest.org/video_index.html) and 3Blue1Brown's channel rather than guessed — a broken or wrong link wastes the exact time this stack is trying to save.

**Part 1 — 3Blue1Brown, "Neural Networks" series (~1 hr, watch in order):**
1. [But what is a neural network? | Deep learning, chapter 1](https://www.youtube.com/watch?v=aircAruvnKk) — what a network actually is: layers, weights, biases, activations. Watch for: this is the literal object LoRA fine-tuning will later modify — everything Sept 18–21 does is "adjust some of these numbers, cheaply."
2. [Gradient descent, how neural networks learn | Deep learning, chapter 2](https://www.youtube.com/watch?v=IHZwWFHWa-w) — the core training loop. Watch for: "the model is bad, and we nudge every weight a little in the direction that makes it less bad" — this is literally what a fine-tuning training run does, just with a LoRA adapter's much smaller set of weights instead of the whole model.
3. [What is backpropagation really doing? | Deep learning, chapter 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U) — the intuition for *how* the nudge in chapter 2 gets computed per-weight.
4. [Backpropagation calculus | Deep learning, chapter 4](https://www.youtube.com/watch?v=tIeHLnjs5U8) — the chain-rule mechanics underneath chapter 3. Optional if short on time — chapters 1–3 carry the intuition Sept 18–21 actually leans on; this one is "how", not "why."

**Part 2 — StatQuest, curated (~2 hrs, any order within the part, but grouped by topic below):**
- Gradient descent, the algorithm itself, both angles: [Gradient Descent, Step-by-Step](https://youtu.be/sDv4f4s2SB8), [Stochastic Gradient Descent, Clearly Explained](https://youtu.be/vMh0zPT0tLI). Watch for: why training uses small batches ("stochastic"), not the whole dataset at once every step — directly explains why a training run reports loss per-step, not just once at the end.
- Loss function: [Neural Networks Part 6: Cross Entropy](https://youtu.be/6ArSys5qHAU). Watch for: this is the specific loss LLM fine-tuning actually minimizes — "how surprised was the model by the correct next token" — the number that goes up or down in a real training log Sept 20.
- Regularization (why LoRA constrains the update instead of retraining everything): [Regularization Part 1: Ridge (L2) Regression](https://youtu.be/Q81RR3yKn30), [Regularization Part 2: Lasso (L1) Regression](https://youtu.be/NGf0voTMlcs). Watch for: the general idea of "penalize big changes to the weights" — LoRA's rank/alpha hyperparameters (Sept 18) are a different mechanism aimed at a similar goal (keep the update small and structured, not free to move every weight arbitrarily).
- [Bias and Variance](https://youtu.be/EuBBz3bI-aA) — the tradeoff under- and over-fitting sit on either side of.
- [Machine Learning Fundamentals: Cross Validation](https://youtu.be/fSytzGwwBVw) — why train/validation/test are three separate sets, not one. Watch for: this is the direct reason Sept 20's training run needs a held-out validation set to actually tell overfitting from genuine learning, rather than just watching training loss go down.

**Total: ~3–4 hrs**, matching the original estimate above.

## Check-in plan (added Sept 15)
Rather than marking this day "done" once the videos are watched (this project's standing habit is verify, don't just mark complete), the close-out for this slot is a short conceptual check-in before Sept 18 starts: explain in your own words what gradient descent is doing, what cross-entropy loss is measuring, why overfitting happens and what regularization does about it, why train/val/test are separate, and the bias-variance tradeoff — using the actual language from the videos, not looked up. Any concept that doesn't come out cleanly gets a targeted mechanism-based explanation (matching your stated preference for that style) before moving on, rather than pushing ahead with a shaky foundation into the LoRA rank/alpha/learning-rate decisions on Sept 18, which are exactly where this would bite.
