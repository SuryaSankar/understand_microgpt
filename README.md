# 🧠 MicroGPT Explained: A Complete Guide for Beginners

This repo is meant to help understand the `microgpt` released by Andrej Karpathy here: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95.

He originally described it as:

> "Train and inference GPT in 200 lines of pure, dependency-free Python. This is the *full* algorithmic content of what is needed. Everything else is just for efficiency. I cannot simplify this any further. The way it works is that the full LLM architecture and loss function is stripped entirely to the most atomic individual mathematical operations that make it up (+, *, **, log, exp), and then a tiny scalar-valued autograd engine (micrograd) calculates gradients. Adam for optim."

All the content in this repo was generated with the help of LLMs only. I am a complete beginner myself and I am trying to dumb it down as much as possible to my level of understanding (which is just high school math basically). The contents in the README were the first attempt by Claude Opus to explain Karpathy's code (cloned as microgpt.py in this repo). But I found even this hard to understand. So I have worked with Claude further to create a detailed line by line explanation of the code in microgpt_explained.py. So you can just head there to read it instead.

---

## Part 1: The Data (Lines 14–29)

```
docs = [l.strip() for l in open('input.txt')...]
```

The code downloads a list of ~32,000 human names (like "Emma", "Olivia", "Liam"). These are its **textbook** — the AI will study them to learn what "name-like" sequences of letters look like.

### The Tokenizer (Lines 23–28)

The AI can't read letters — it only understands numbers. So we build a **dictionary** mapping each unique character to a number:

| Character | Token ID |
|-----------|----------|
| `a` | 0 |
| `b` | 1 |
| ... | ... |
| `z` | 25 |
| `BOS` | 26 |

`BOS` means **"Beginning/End of Sequence"** — a special marker saying "the name starts/ends here." Think of it like a period at the end of a sentence.

So the name `"cat"` might become `[26, 2, 0, 19, 26]` — BOS, c, a, t, BOS.

---

## Part 2: The Autograd Engine — The `Value` Class (Lines 31–65)

This is the **heart** of the entire file. It's the engine that lets the AI **learn**.

### What Problem Does It Solve?

Imagine you have a math formula: $y = 3x^2 + 2x$. If $x = 4$, then $y = 56$. But what if you want to know: **"If I nudge $x$ a tiny bit, how much does $y$ change?"** That's the **derivative** (or **gradient**):

$$\frac{dy}{dx} = 6x + 2 = 26 \quad \text{(when } x=4 \text{)}$$

This means: if you increase $x$ by 0.001, $y$ increases by about $26 \times 0.001 = 0.026$.

**Why does the AI care?** Because learning = adjusting numbers to make mistakes smaller. Gradients tell you **which direction** to adjust each number and **by how much**.

### How `Value` Works

Each `Value` wraps a single number and remembers:
- **`data`**: the actual number (e.g., `3.14`)
- **`grad`**: how much the final loss changes if this number changes (starts at 0, filled in during `backward()`)
- **`_children`**: what values were combined to create this one
- **`_local_grads`**: the derivative of this operation w.r.t. each child

### The Math Operations

Each operation records the **calculus rule** for its derivative:

| Operation | Forward (compute result) | Local Gradient (derivative rule) |
|-----------|-------------------------|----------------------------------|
| $a + b$ | $a + b$ | $\frac{\partial}{\partial a} = 1, \quad \frac{\partial}{\partial b} = 1$ |
| $a \times b$ | $a \times b$ | $\frac{\partial}{\partial a} = b, \quad \frac{\partial}{\partial b} = a$ |
| $a^n$ | $a^n$ | $n \cdot a^{n-1}$ |
| $\log(a)$ | $\ln(a)$ | $\frac{1}{a}$ |
| $e^a$ | $e^a$ | $e^a$ |
| $\text{relu}(a)$ | $\max(0, a)$ | $1$ if $a > 0$, else $0$ |

**Example**: If $c = a \times b$, and $a = 3, b = 5$, then:
- `c.data = 15`
- The local gradient w.r.t. $a$ is $b = 5$ (if $a$ increases by 1, $c$ increases by 5)
- The local gradient w.r.t. $b$ is $a = 3$

### The `backward()` Method — Chain Rule

This is where the magic happens. The **chain rule** from calculus says:

$$\frac{\partial \text{loss}}{\partial x} = \frac{\partial \text{loss}}{\partial y} \times \frac{\partial y}{\partial x}$$

In plain English: *"To find how $x$ affects the loss, multiply how $y$ affects the loss by how $x$ affects $y$."*

The `backward()` method:
1. **Sorts** all the operations in order (topological sort) — like a recipe's steps in order
2. **Starts** at the loss (the final answer) with gradient = 1
3. **Walks backward** through every operation, using the chain rule to propagate gradients to every single parameter

Think of it like **blame assignment**: if your cake tastes bad (high loss), backward tells you exactly how much to blame each ingredient (parameter).

---

## Part 3: Model Parameters (Lines 67–80)

```python
n_embd = 16      # each token is represented by 16 numbers
n_head = 4       # 4 "attention heads" (more on this below)
n_layer = 1      # 1 layer of processing
block_size = 16   # max name length the model can handle
```

The **parameters** are thousands of numbers (initialized randomly) that the model will learn. They're organized into **matrices** (grids of numbers):

- **`wte`** (word token embedding): a lookup table — each token ID maps to a row of 16 numbers. This is the AI's way of representing each letter as a point in 16-dimensional space.
- **`wpe`** (word position embedding): same idea, but for **position** (1st letter, 2nd letter, etc.).
- **`attn_wq, wk, wv, wo`**: attention matrices (explained below).
- **`mlp_fc1, fc2`**: the "thinking" matrices.
- **`lm_head`**: converts the model's internal representation back to predictions over characters.

Total: **~11,000 numbers** that start random and get tuned by training.

---

## Part 4: The Model Architecture (Lines 82–126)

This is the actual GPT. It's a **function**: give it a letter and its position → it predicts what the next letter should be.

### `linear(x, w)` — Matrix Multiplication

$$\text{output}_i = \sum_j w_{ij} \cdot x_j$$

This is like a weighted vote. Each output is a **weighted sum** of all inputs. The weights $w$ are learnable parameters. This is the fundamental building block of neural networks — it lets the model **mix information**.

**Analogy**: Imagine 16 friends each give you a rating (the input `x`). You trust some friends more than others (the weights `w`). The linear function computes your overall opinion by weighting each friend's rating.

### `softmax(logits)` — Turn Scores Into Probabilities

$$P_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

This takes a list of raw scores (which can be any number, positive or negative) and converts them into **probabilities** that sum to 1.

**Example**: Scores `[2.0, 1.0, 0.1]` → Probabilities `[0.66, 0.24, 0.10]`

The highest score gets the highest probability, but lower scores still get a chance. The exponential $e^z$ ensures all values are positive.

### `rmsnorm(x)` — Normalization

$$\text{rmsnorm}(x_i) = \frac{x_i}{\sqrt{\frac{1}{n}\sum_j x_j^2 + \epsilon}}$$

This **normalizes** the numbers so they don't explode to billions or shrink to near-zero as they pass through many operations. Think of it as keeping the volume at a reasonable level — not too loud, not too quiet.

### `gpt(token_id, pos_id, keys, values)` — The Main Model

Here's what happens step by step when the model processes one letter:

#### Step 1: Embeddings (Lines 99–102)
```python
tok_emb = state_dict['wte'][token_id]  # look up the letter's representation
pos_emb = state_dict['wpe'][pos_id]    # look up the position's representation
x = [t + p for t, p in zip(tok_emb, pos_emb)]  # combine them
```

The letter "a" in position 3 gets a different representation than "a" in position 7, because position matters! "ca**t**" and "ta**c**" have the same letters but different meanings.

#### Step 2: Multi-Head Attention (Lines 106–120)

This is the most famous part of GPT. **Attention** answers the question: *"When predicting the next letter, which previous letters should I pay attention to?"*

Here's the analogy. Imagine you're trying to guess the next letter in "Mich___":

1. **Query (Q)**: The current letter asks "What am I looking for?" — like raising your hand with a question.
2. **Key (K)**: Each previous letter advertises "Here's what I contain" — like a label on a folder.
3. **Value (V)**: Each previous letter also says "Here's the useful info I have" — the actual content of the folder.

The math:

$$\text{attention score} = \frac{Q \cdot K}{\sqrt{d}}$$

This is a **dot product** — it measures how similar the query and key are. Divide by $\sqrt{d}$ to keep numbers stable. Then softmax turns scores into weights:

$$\text{output} = \sum_t \text{weight}_t \times V_t$$

**Multi-head** means we do this 4 times in parallel (with different Q, K, V weights), each looking at a different 4-dimensional slice. It's like having 4 different experts, each paying attention to different patterns (one might focus on vowels, another on consonant clusters, etc.).

#### Step 3: MLP Block (Lines 122–126)

```python
x = linear(x, mlp_fc1)   # expand: 16 → 64 dimensions
x = [xi.relu() for xi in x]  # ReLU activation
x = linear(x, mlp_fc2)   # compress: 64 → 16 dimensions
```

After attention gathers relevant info, the **MLP (Multi-Layer Perceptron)** does the actual "thinking." It's two linear transformations with a **ReLU** in between.

**ReLU** is dead simple: $\text{relu}(x) = \max(0, x)$. If the number is negative, make it zero. If positive, keep it. This introduces **non-linearity** — without it, stacking linear functions would just give you another linear function (you couldn't learn complex patterns).

**Residual connections** (`x = [a + b for a, b in zip(x, x_residual)]`): The output is **added** to the input. This is like saying "start with what you had, then add what you learned." It prevents the signal from degrading in deeper networks.

#### Step 4: Output
```python
logits = linear(x, state_dict['lm_head'])
```

Finally, the 16-dimensional representation is projected to **27 scores** (one per possible next character + BOS). These scores go through softmax to become probabilities.

---

## Part 5: Training Loop (Lines 129–159)

This is where the AI **learns** by repeatedly:

### 1. Forward Pass — Make a Prediction

For the name "Emma":
- Feed in `BOS` → predict `E` (probably wrong at first)
- Feed in `E` → predict `m`
- Feed in `m` → predict `m`
- Feed in `m` → predict `a`
- Feed in `a` → predict `BOS` (end)

### 2. Compute Loss — How Wrong Were We?

$$\text{loss} = -\log(P_{\text{correct}})$$

If the model gives the correct next letter a probability of 0.9 → loss = $-\log(0.9) = 0.105$ (small, good!)
If the model gives it a probability of 0.01 → loss = $-\log(0.01) = 4.6$ (big, bad!)

This is called **cross-entropy loss**. It heavily punishes confident wrong answers.

### 3. Backward Pass — Find Who's to Blame

`loss.backward()` — this runs the chain rule through the entire computation graph, computing the gradient for every one of the ~11,000 parameters.

### 4. Adam Optimizer — Update Parameters

```python
m[i] = β₁ · m[i] + (1 - β₁) · gradient        # momentum (smoothed gradient)
v[i] = β₂ · v[i] + (1 - β₂) · gradient²        # velocity (smoothed squared gradient)
parameter -= lr · m̂ / (√v̂ + ε)                  # update step
```

**Plain gradient descent** would be: `parameter -= learning_rate * gradient`. But Adam is smarter:

- **Momentum ($m$)**: Instead of following today's gradient exactly, it keeps a running average. Like a ball rolling downhill — it builds up speed in a consistent direction and doesn't jerk around with every bump.
- **Adaptive rate ($v$)**: Parameters with consistently large gradients get smaller steps; those with small gradients get bigger steps. Everyone learns at their own pace.
- **Learning rate decay**: `lr_t = lr * (1 - step/num_steps)` — start with big steps, take smaller steps as you get closer to the answer. Like using a coarse sandpaper first, then fine sandpaper.

---

## Part 6: Inference — Generate New Names (Lines 162–175)

After training, the model can **hallucinate** new names:

1. Start with `BOS`
2. Get probabilities for the next letter
3. **Randomly sample** a letter (weighted by probability)
4. Feed that letter back in, repeat
5. Stop when it produces `BOS` (end marker)

**Temperature** controls randomness:
- Temperature = 0.1: Almost always picks the highest-probability letter → boring, repetitive
- Temperature = 1.0: Samples proportionally → creative, sometimes weird
- Temperature = 0.5 (used here): A balance between the two

Mathematically, dividing logits by temperature before softmax sharpens or flattens the distribution:

$$P_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

---

## The Big Picture 🎯

| Stage | What Happens | Analogy |
|-------|-------------|---------|
| **Tokenize** | Convert letters → numbers | Translating English to Morse code |
| **Embed** | Numbers → rich vectors | Looking up a word in a thesaurus |
| **Attention** | "Which previous letters matter?" | Reading comprehension |
| **MLP** | "What pattern do I see?" | Thinking and reasoning |
| **Softmax** | Scores → probabilities | Ranking your guesses |
| **Loss** | Measure how wrong we were | Getting your test score |
| **Backward** | Figure out who's to blame | Going over wrong answers |
| **Adam** | Adjust all parameters | Studying for next time |
| **Repeat 1000×** | Keep improving | Practice makes perfect |

The remarkable thing about this code is that the **entire intelligence** — the ability to learn patterns in names and generate new ones — emerges from just six atomic math operations: `+`, `×`, `**`, `log`, `exp`, and `max`. Everything else is just these operations composed together in clever ways, with gradients flowing backward to improve 11,000 numbers.
