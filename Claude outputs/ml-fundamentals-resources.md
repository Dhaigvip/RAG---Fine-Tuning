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
