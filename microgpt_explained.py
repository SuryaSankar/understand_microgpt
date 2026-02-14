"""
The most atomic way to train and inference a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy
"""

import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42) # Let there be order among chaos
                # Sets the random number generator to a fixed starting point.
                # This makes the code REPRODUCIBLE — every time you run it, the
                # random weight initialization, data shuffling, and sampling will
                # produce the exact same results. Change 42 to any other number
                # and you'll get a different (but still reproducible) run.

# ==========================================================================
# Let there be an input dataset `docs`: list[str] of documents.
#
# The TRAINING DATA — what the model learns from. Every neural network needs
# data to learn patterns from, and this model's data is a list of ~32,000
# human names (e.g., "Emma", "Olivia", "Liam", "Noah", ...).
#
# The file comes from Karpathy's "makemore" project — a dataset of the most
# common baby names in the US. Each line in the file is one name.
#
# Why names? They're a perfect toy dataset because:
#   - They're short (most are 3-10 characters) → fast to train on
#   - They have clear patterns (common endings like -lyn, -son, -ella)
#   - It's easy to tell if generated names look realistic
# ==========================================================================
if not os.path.exists('input.txt'):
    # Download the dataset if it doesn't already exist locally.
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

# Read the file, split it into individual lines, strip whitespace, and filter
# out any empty lines. The result is a plain Python list of strings:
#   docs = ["Emma", "Olivia", "Ava", "Isabella", ...]
# Each string is one "document" — in this case, one name.
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]

# Shuffle the names into a random order before training.
# Why? During training, the model processes names one at a time (step 0 gets
# docs[0], step 1 gets docs[1], etc.). If the names were in alphabetical order,
# the model would see all the A-names first, then all the B-names, etc.
# This creates a problem: the model would temporarily "specialize" in whatever
# letter group it's currently seeing, then forget it when the next group starts.
# Shuffling ensures each training step sees a random name, so the model learns
# general patterns about ALL names simultaneously rather than overfitting to
# one region of the alphabet at a time. This is standard practice in ML.
random.shuffle(docs)
print(f"num docs: {len(docs)}")


# Let there be a Tokenizer to translate strings to discrete symbols and back.

# ==========================================================================
# Neural networks can't process raw text — they only understand numbers.
# A tokenizer converts text into numbers (encoding) and back (decoding).
#
# This is the SIMPLEST possible tokenizer: character-level. Each unique
# letter in the dataset gets its own integer ID. More sophisticated
# tokenizers (like GPT's BPE) work with sub-word chunks like "ing" or
# "tion", but for short names, individual characters work perfectly.
# ==========================================================================

# uchars: the "vocabulary" — a sorted list of every unique character found
# across all names in the dataset.
#
# How it works, step by step:
#   ''.join(docs)   → glue all names into one giant string: "EmmaOliviaAva..."
#   set(...)         → remove duplicates, leaving just unique chars: {'E','m','a','O',...}
#   sorted(...)      → sort alphabetically: ['a','b','c',...,'x','y','z']
#
# The POSITION of each character in this list becomes its token ID:
#   uchars[0] = 'a' → token ID 0
#   uchars[1] = 'b' → token ID 1
#   ...
#   uchars[25] = 'z' → token ID 25
#
# So encoding is: char → uchars.index(char) → integer
# And decoding is: integer → uchars[integer] → char
# For the names dataset, uchars contains all 26 lowercase letters.
uchars = sorted(set(''.join(docs)))

# BOS: the "Beginning of Sequence" token — a special marker that doesn't
# represent any real letter. It serves TWO purposes:
#
#   1. START signal: placed at the beginning of every name during training.
#      It tells the model "a new name is starting — predict the first letter."
#      Without it, the model wouldn't know how to begin generating.
#
#   2. STOP signal: placed at the end of every name during training.
#      The model learns that after the last letter, BOS comes next.
#      During generation, when the model predicts BOS, we know the name is done.
#
# Its ID is 26 (the next number after the 26 letters: 0-25).
# Real LLMs have similar special tokens: <|endoftext|> in GPT, <s> in LLaMA, etc.
BOS = len(uchars)

# vocab_size: the total number of distinct tokens the model needs to handle.
# 26 letters + 1 BOS = 27 tokens.
#
# This number determines the size of key model components:
#   - wte (token embedding table):  27 rows — one embedding vector per token
#   - lm_head (output projection):  27 rows — one score per possible next token
#   - softmax output:               27 probabilities — one per possible next token
#
# Every token ID in the system is a number from 0 to vocab_size-1 (i.e., 0 to 26).
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")


# Let there be Autograd, to recursively apply the chain rule through a computation graph

# ==========================================================================
# "Autograd" = AUTOmatic GRADient computation. In machine learning, we need
# to compute gradients (derivatives) of the loss w.r.t. every parameter —
# that's how the model knows which direction to adjust each weight. Doing
# this by hand for thousands of parameters would be impossible. Autograd
# does it automatically: you just write the forward math normally, and
# it records every operation behind the scenes so it can "replay" them
# backward to compute all the gradients at once (via the chain rule).
# PyTorch, TensorFlow, and JAX all have autograd engines built in.
# This code builds one from scratch using the Value class below.
#
# The Value class is the HEART of this entire system. It is a "smart number"
# wrapper — every number in the model (every weight, every intermediate
# calculation) is wrapped in a Value object instead of being a plain float.
#
# Why? Because Value does TWO things at once:
#   1. FORWARD PASS: computes the actual math (2 + 3 = 5, like normal numbers)
#   2. BACKWARD PASS: remembers HOW the result was computed, so we can later
#      figure out the gradient (derivative) — i.e., "if I wiggle this input
#      a tiny bit, how much does the final loss change?"
#
# Together, the Value objects form a COMPUTATION GRAPH — a tree of operations
# that connects every model parameter to the final loss. During training,
# we walk backward through this graph (backpropagation) to compute gradients,
# which tell us how to adjust each parameter to reduce the loss.
#
# Every math operation (+, *, **, log, exp, relu) is overloaded to:
#   (a) compute the result (forward pass)
#   (b) record the children (inputs to this operation)
#   (c) record the local gradient (derivative of this operation w.r.t. each input)
# Later, backward() uses (b) and (c) to propagate gradients via the chain rule.
# ==========================================================================
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    # __slots__ is a Python optimization: it tells Python "these are the ONLY
    # attributes this class will ever have." This saves memory because Python
    # won't create a __dict__ for each instance. Since we create ~100,000+
    # Value objects during a single forward pass, this adds up!

    def __init__(self, data, children=(), local_grads=()):
        # data: the actual numerical value of this node (a plain float).
        #       During the forward pass, this is computed from the children's data.
        #       e.g., if this Value is the result of 3 + 5, data = 8.
        self.data = data

        # grad: the gradient — how much the FINAL LOSS changes if we nudge this
        #       value by a tiny amount. Starts at 0 and gets filled in during
        #       backward(). For a model parameter, the gradient tells us which
        #       direction to adjust it to reduce the loss.
        #       e.g., grad = -0.3 means "increasing this value slightly would
        #       decrease the loss by 0.3 × (the nudge amount)."
        self.grad = 0

        # _children: a tuple of the Value objects that were INPUTS to the
        #            operation that created this Value. These are the edges in
        #            the computation graph pointing backward (toward the inputs).
        #            e.g., if c = a + b, then c._children = (a, b).
        #            Leaf nodes (model parameters, constants) have no children: ().
        self._children = children

        # _local_grads: a tuple of the LOCAL DERIVATIVES — the derivative of
        #               THIS operation's output w.r.t. each child, evaluated at
        #               the current input values.
        #               e.g., for c = a + b: local_grads = (1, 1)
        #                     because ∂(a+b)/∂a = 1 and ∂(a+b)/∂b = 1.
        #               e.g., for c = a * b where a=3, b=5: local_grads = (5, 3)
        #                     because ∂(a*b)/∂a = b = 5 and ∂(a*b)/∂b = a = 3.
        #               These are used by backward() in the chain rule:
        #                     child.grad += local_grad * parent.grad
        self._local_grads = local_grads

    # ======================================================================
    # ARITHMETIC OPERATIONS — each one creates a new Value node in the graph
    #
    # Pattern for every operation:
    #   Value(result, children=(inputs...), local_grads=(derivatives...))
    #
    # The "local_grads" are the CALCULUS DERIVATIVES of the operation, which
    # you may remember from high school:
    #   - d/dx (x + y) = 1           (adding: both inputs contribute equally)
    #   - d/dx (x * y) = y           (multiplying: derivative w.r.t. x is y)
    #   - d/dx (x^n)   = n * x^(n-1) (power rule)
    #   - d/dx (ln x)  = 1/x         (log derivative)
    #   - d/dx (e^x)   = e^x         (exp is its own derivative!)
    # ======================================================================

    def __add__(self, other):
        # Addition: c = a + b
        # Ensures the other operand is a Value (wraps plain numbers like 3.0).
        other = other if isinstance(other, Value) else Value(other)
        # Forward:  c.data = a.data + b.data
        # Children: (a, b)
        # Local gradients: (1, 1)
        #   ∂(a+b)/∂a = 1  — if you increase a by 1, the sum increases by 1.
        #   ∂(a+b)/∂b = 1  — same for b. Both contribute equally.
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        # Multiplication: c = a * b
        other = other if isinstance(other, Value) else Value(other)
        # Forward:  c.data = a.data * b.data
        # Children: (a, b)
        # Local gradients: (b.data, a.data)
        #   ∂(a*b)/∂a = b  — if you increase a by 1, the product increases by b.
        #   ∂(a*b)/∂b = a  — if you increase b by 1, the product increases by a.
        #   e.g., if a=3, b=5: d/da(3*5) = 5, d/db(3*5) = 3.
        #   This makes intuitive sense: in 3×5, increasing 3 by 1 gives 4×5=20,
        #   a change of 5 (which is the value of b).
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        # Power: c = a^n (where n is a plain number, NOT a Value)
        # Forward:  c.data = a.data ** n
        # Children: (a,) — only one child, since n is a constant
        # Local gradient: n * a^(n-1)  — the classic POWER RULE from calculus.
        #   e.g., d/dx(x³) = 3x². If x=2: gradient = 3*4 = 12.
        #   This is used for operations like x**2 (squaring in rmsnorm),
        #   x**-1 (division: a/b = a * b**-1), and x**-0.5 (inverse sqrt).
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        # Natural logarithm: c = ln(a)
        # Forward:  c.data = ln(a.data)
        # Children: (a,)
        # Local gradient: 1 / a.data
        #   d/dx(ln(x)) = 1/x. If x=2: gradient = 0.5.
        #   Intuitively: the larger x is, the flatter the log curve, so the
        #   derivative gets smaller. For small x near 0, the curve is steep
        #   so the derivative is huge (1/0.01 = 100).
        #   Used in the cross-entropy loss: -log(probability of correct token).
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self):
        # Exponential: c = e^a
        # Forward:  c.data = e^(a.data)
        # Children: (a,)
        # Local gradient: e^(a.data)  — exp is the only function that is its
        #   own derivative! d/dx(e^x) = e^x.
        #   If x=2: e^2 ≈ 7.39, and the gradient is also 7.39.
        #   Used in softmax: softmax converts logits to probabilities using exp.
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        # ReLU (Rectified Linear Unit): c = max(0, a)
        # The simplest activation function: passes positive values through
        # unchanged, and clamps negative values to zero.
        # Forward:  c.data = max(0, a.data)
        # Children: (a,)
        # Local gradient: 1.0 if a.data > 0, else 0.0
        #   If the input was positive, the output = input, so gradient = 1
        #   (changes pass straight through).
        #   If the input was negative, the output = 0 regardless, so gradient = 0
        #   (changes are blocked — this neuron is "dead" for this input).
        #   e.g., relu(3) = 3, gradient = 1. relu(-2) = 0, gradient = 0.
        #   Used in the MLP block to introduce non-linearity.
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        # Goal: compute how much the final loss changes if we nudge each Value
        # in the computation graph. This is done via the chain rule, walking
        # backward from the loss to every parameter.

        # Step 1: Build a topological ordering of the computation graph.
        # "Topological order" means: every node appears AFTER all of its children.
        # This is like listing recipe steps so that ingredients come before the
        # dish that uses them. We need this order so that when we walk backward,
        # a node's gradient is fully accumulated before we propagate through it.
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)  # recurse into children first (depth-first)
                topo.append(v)        # then append this node (so it comes after children)
        build_topo(self)

        # Step 2: The loss's gradient w.r.t. itself is 1.
        # "How much does the loss change if the loss changes by 1?" → exactly 1.
        # This is the starting point for the chain rule.
        self.grad = 1

        # Step 3: Walk through nodes in REVERSE topological order (from loss
        # back toward the inputs/parameters). For each node, propagate its
        # gradient to its children using the chain rule:
        #
        #   child.grad += local_grad * parent.grad
        #
        # where local_grad is the derivative of the parent w.r.t. this child
        # (computed during the forward pass), and parent.grad is how much the
        # loss changes w.r.t. the parent (already computed by this point).
        #
        # We use += because a child may contribute to multiple parents,
        # and all those contributions to the gradient must be summed up.
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# Initialize the parameters, to store the knowledge of the model.
# --------------------------------------------------------------------------
# Hyperparameters: these are the "knobs" that control the size of the model.
# They are set intentionally small here because this code uses a pure-Python
# scalar autograd engine (no GPU, no NumPy), so training is very slow.
# Real GPTs use values 10-100x larger.
# --------------------------------------------------------------------------
n_embd = 16     # Embedding dimension: each token (letter) is represented as a
                # vector of 16 numbers. This is the "width" of the model — how
                # much information each token carries. GPT-2 uses 768; here 16
                # is enough to learn simple letter patterns in names.

n_head = 4      # Number of attention heads: attention is split into 4 independent
                # "experts", each working on a 4-dimensional slice (16 / 4 = 4).
                # Each head can learn to focus on different patterns (e.g. one head
                # might track vowels, another consonant pairs). GPT-2 uses 12.

n_layer = 1     # Number of transformer layers: how many times the data passes
                # through an attention + MLP block. Each layer refines the
                # representation further. Set to 1 here because (a) names are
                # short and simple, and (b) more layers = proportionally slower
                # training in this pure-Python implementation. GPT-2 uses 12-48.
                # You could change this to 2 or 3 and the code would still work,
                # just slower, with more parameters to learn.

block_size = 16 # Maximum sequence length: the longest name the model can process
                # (in characters). Names longer than 16 chars get truncated.
                # This also determines the size of the position embedding table
                # (wpe), since each position 0..15 gets its own learned vector.
                # GPT-2 uses 1024; 16 is plenty since most names are under 15 chars.

head_dim = n_embd // n_head # Dimension of each attention head (16 / 4 = 4).
                            # The embedding is split evenly across heads, so each
                            # head works with a smaller slice of the full vector.
                            # This keeps the total computation the same whether you
                            # use 1 head of size 16 or 4 heads of size 4.

# Helper to create a matrix (2D list) of random Value objects.
# - nout × nin: the matrix has nout rows and nin columns.
# - Each entry is a Value initialized with a small random number drawn from a
#   Gaussian (bell curve) distribution centered at 0, with std dev = 0.08.
# - Why small random numbers? If they started at 0, every neuron would be identical
#   and learn the same thing. If they started too large, the math would explode
#   (huge numbers flowing through exp, log, etc.). Small random values break
#   symmetry while keeping things numerically stable.
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

# state_dict holds ALL learnable parameters of the model, organized by name.
# Think of it as the model's "brain" — thousands of numbers that start random
# and get tuned during training to encode knowledge about name patterns.
#
# The three "global" parameter matrices:
#   wte  (word token embedding):    vocab_size × n_embd  (27 × 16 = 432 params)
#        A lookup table: each of the 27 tokens (26 letters + BOS) gets its own
#        row of 16 numbers. This is how the model represents each letter as a
#        point in 16-dimensional space. Letters that behave similarly (like
#        vowels) will end up with similar vectors after training.
#
#   wpe  (word position embedding): block_size × n_embd  (16 × 16 = 256 params)
#        Same idea, but for positions. Position 0 gets one vector, position 1
#        gets another, etc. This lets the model know WHERE in the name a letter
#        appears — "a" at the start vs "a" at the end should be treated differently.
#
#   lm_head (language model head):  vocab_size × n_embd  (27 × 16 = 432 params)
#        The output projection: takes the model's final 16-dim representation
#        and produces 27 scores (one per possible next token). The highest score
#        becomes the model's prediction for what letter comes next.
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}

# Per-layer parameters: each transformer layer gets its own set of weight matrices.
# With n_layer=1, this loop runs once, creating one set.
for i in range(n_layer):
    # Attention block matrices (each n_embd × n_embd = 16 × 16 = 256 params):
    #   attn_wq: transforms input into "queries"  — "what am I looking for?"
    #   attn_wk: transforms input into "keys"     — "what do I contain?"
    #   attn_wv: transforms input into "values"   — "what info do I carry?"
    #   attn_wo: combines all attention heads' outputs back into one vector
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    # MLP block matrices:
    #   mlp_fc1: expands from 16 → 64 dimensions (4× expansion ratio)
    #            This wider intermediate layer gives the network more room
    #            to represent complex patterns before compressing back down.
    #            (4*n_embd × n_embd = 64 × 16 = 1024 params)
    #   mlp_fc2: compresses from 64 → 16 dimensions back to embedding size
    #            (n_embd × 4*n_embd = 16 × 64 = 1024 params)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

# Flatten all parameters into a single 1D list of Value objects.
# The parameters currently live inside state_dict as nested structures:
#   state_dict → multiple matrices → each matrix has rows → each row has Values
# The optimizer (Adam) needs to loop over EVERY individual parameter to update it.
# Rather than writing nested loops every time, we flatten them once into a simple
# list: [Value, Value, Value, ...]. Now the optimizer can just do:
#   for i, p in enumerate(params): ...update p...
# This also makes it easy to create matching optimizer buffers (m and v) of the
# same length — one momentum and one velocity value per parameter.
params = [p for mat in state_dict.values() for row in mat for p in row]
#         ↑ for each matrix in state_dict   ↑ for each row   ↑ for each Value
print(f"num params: {len(params)}")

# --------------------------------------------------------------------------
# Define the model architecture.
#
# The model is a STATELESS FUNCTION: given a token (letter) and its position,
# it returns "logits" — raw scores for what the next token should be.
#
# What are logits?
#   Logits are unnormalized scores — one number per possible next token (27 in
#   our case: 26 letters + BOS). They can be any value: positive, negative, or
#   zero. A higher logit means the model thinks that token is more likely.
#   Example: logits might be [2.1, -0.5, 0.3, ...] for [a, b, c, ...].
#   These raw scores are then passed through softmax to become probabilities
#   that sum to 1.0: [0.52, 0.04, 0.09, ...].
#   The name "logit" comes from being the inverse of the logistic (sigmoid)
#   function — they live in "log-odds" space before being converted to probs.
#
# This architecture follows GPT-2, with three simplifications:
#   1. LayerNorm → RMSNorm: a simpler normalization that skips the mean-centering
#      step. RMSNorm only divides by the root-mean-square, which works just as
#      well in practice and is easier to implement.
#   2. No biases: GPT-2 adds a bias term (b) to every linear layer: y = Wx + b.
#      This code omits biases entirely. They add parameters but aren't essential.
#   3. GeLU → ReLU: GPT-2 uses GeLU (a smooth activation), but ReLU (simply
#      max(0, x)) is simpler and works fine for this toy model.
# --------------------------------------------------------------------------

def linear(x, w):
    # Matrix-vector multiplication: the fundamental building block of neural networks.
    #
    # Inputs:
    #   x: a vector (list) of Values, e.g. [x0, x1, ..., x15] (length n_in)
    #   w: a matrix (list of lists) of Values, with shape n_out × n_in
    #
    # For each row `wo` in the weight matrix, compute the DOT PRODUCT with x:
    #   output_i = wo[0]*x[0] + wo[1]*x[1] + ... + wo[n_in-1]*x[n_in-1]
    #
    # This produces one output value per row, so the result has length n_out.
    #
    # Analogy: imagine x is a list of 16 ingredient amounts, and each row of w
    # is a recipe that says how much of each ingredient to use. Each output is
    # one dish — a weighted combination of all ingredients. The weights are
    # learnable, so the model learns WHICH combinations of inputs matter.
    #
    # Example: if x = [a, b] and w = [[2, 3], [4, 5]], then:
    #   output = [2a + 3b, 4a + 5b]  — two different weighted sums of the same inputs.
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    # Convert raw scores (logits) into probabilities that sum to 1.
    #
    # Formula: P_i = exp(z_i) / sum(exp(z_j) for all j)
    #
    # Step 1: Find the maximum logit value. We subtract it from all logits before
    # taking exp(). This is a numerical stability trick — it doesn't change the
    # result mathematically (the max cancels out in the division), but prevents
    # exp() from producing astronomically large numbers that could overflow.
    # e.g., exp(1000) would overflow, but exp(1000 - 1000) = exp(0) = 1 is fine.
    max_val = max(val.data for val in logits)
    # Step 2: Compute exp(logit - max) for each logit. The exponential ensures
    # all values are positive (you can't have negative probability). Larger
    # logits produce exponentially larger values, so the highest-scored token
    # dominates.
    exps = [(val - max_val).exp() for val in logits]
    # Step 3: Sum all the exponentials — this becomes the denominator.
    total = sum(exps)
    # Step 4: Divide each exponential by the total so everything sums to 1.
    # Example: logits [2.0, 1.0, 0.1] → exps [7.39, 2.72, 1.10] → total 11.21
    #          → probs [0.66, 0.24, 0.10] — they sum to 1.0!
    return [e / total for e in exps]

def rmsnorm(x):
    # Root Mean Square Normalization: keeps numbers in a healthy range.
    #
    # Without normalization, values can grow huge or shrink to near-zero as they
    # pass through many multiplications. This makes training unstable — gradients
    # either explode or vanish. RMSNorm rescales the vector so its "average
    # magnitude" is approximately 1.
    #
    # Step 1: Compute the mean of squared values.
    # For x = [3, 4], ms = (9 + 16) / 2 = 12.5
    ms = sum(xi * xi for xi in x) / len(x)
    # Step 2: Compute the scaling factor = 1 / sqrt(ms + epsilon).
    # The epsilon (1e-5 = 0.00001) prevents division by zero if all values are 0.
    # The ** -0.5 is the same as 1 / sqrt(...).
    # For our example: scale = 1 / sqrt(12.5) ≈ 0.283
    scale = (ms + 1e-5) ** -0.5
    # Step 3: Multiply each element by the scale factor.
    # [3, 4] * 0.283 = [0.849, 1.131] — now the values are in a reasonable range.
    # The relative proportions between elements are preserved (3:4 ratio stays),
    # but the overall magnitude is normalized.
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    # The GPT model function: processes ONE token at a time and returns logits
    # (scores) predicting what the next token should be.
    #
    # Inputs:
    #   token_id: integer ID of the current token (e.g., 0 for 'a', 26 for BOS)
    #   pos_id:   integer position in the sequence (0 = first char, 1 = second, etc.)
    #   keys:     a list of lists, one per layer. Each stores the "key" vectors
    #             from ALL previous tokens processed so far. This is the "KV cache"
    #             — it lets the model remember what it has already seen without
    #             recomputing everything from scratch each time.
    #   values:   same structure as keys, but storing "value" vectors from past tokens.
    #
    # Output:
    #   logits: a list of 27 raw scores (one per possible next token). Higher score
    #           = model thinks that token is more likely to come next.

    # --- Step 1: Look up embeddings ---
    # Retrieve the 16-number vector for this token from the embedding table.
    # e.g., if token_id=0 (letter 'a'), tok_emb = wte[0] = [Value, Value, ..., Value]
    tok_emb = state_dict['wte'][token_id] # token embedding
    # Retrieve the 16-number vector for this position.
    # e.g., if pos_id=3, pos_emb = wpe[3] = [Value, Value, ..., Value]
    pos_emb = state_dict['wpe'][pos_id] # position embedding
    # Add them element-wise: x[i] = tok_emb[i] + pos_emb[i].
    # This combines "what letter is this?" with "where in the name is it?"
    # so that 'a' at position 0 has a different representation than 'a' at position 5.
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
    # Normalize x so the magnitudes don't blow up before entering the layers.
    x = rmsnorm(x)

    # --- Step 2: Pass through each transformer layer ---
    # Each layer has two sub-blocks: (1) attention and (2) MLP.
    # With n_layer=1, this loop runs once.
    for li in range(n_layer):

        # =====================================================================
        # 1) MULTI-HEAD ATTENTION BLOCK
        # Purpose: let the current token "look back" at all previous tokens
        # and gather relevant information from them.
        # =====================================================================

        # Save the current x so we can add it back later (residual connection).
        # This is like saying "remember where I started, so I can add what I
        # learn on top of it" — prevents information loss in deep networks.
        x_residual = x
        # Normalize before attention (pre-norm architecture).
        x = rmsnorm(x)

        # Project x into three different 16-dimensional vectors using learned weights:
        #   q (query):  "What am I looking for?" — the current token's question
        #   k (key):    "What do I contain?" — this token's label/advertisement
        #   v (value):  "What useful info do I carry?" — this token's actual content
        # Each is computed as: linear(x, W) = W @ x (matrix-vector multiply)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])

        # Store this token's key and value in the KV cache so FUTURE tokens can
        # attend to this one. keys[li] and values[li] grow by one entry each
        # time gpt() is called for a new position.
        keys[li].append(k)
        values[li].append(v)

        # x_attn will accumulate the output from all attention heads,
        # concatenated together. Each head contributes head_dim=4 values,
        # and 4 heads × 4 values = 16 total (matching n_embd).
        x_attn = []

        # --- Multi-head: run 4 independent attention heads ---
        for h in range(n_head):
            # Each head works on its own slice of the 16-dim vectors.
            # Head 0 uses dims [0:4], head 1 uses [4:8], head 2 uses [8:12], head 3 uses [12:16].
            hs = h * head_dim  # head start index

            # Extract this head's 4-dimensional slice of the query.
            q_h = q[hs:hs+head_dim]
            # Extract this head's slice from ALL stored keys (from every past token).
            # k_h is a list of 4-dim vectors, one per past token.
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            # Same for values.
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]

            # Compute attention scores: how relevant is each past token to the
            # current query? This is the dot product of q_h with each key,
            # divided by sqrt(head_dim) for numerical stability.
            #
            # For each past token t:
            #   score_t = (q_h[0]*k_h[t][0] + q_h[1]*k_h[t][1] + ... ) / sqrt(4)
            #
            # High dot product = query and key point in similar directions = relevant.
            # The / sqrt(head_dim) prevents dot products from growing too large
            # (which would make softmax produce near-0 and near-1 values, killing
            # gradients — known as the "attention score scaling" trick).
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]

            # Convert raw attention scores into weights that sum to 1.
            # e.g., scores [3.2, 1.1, 0.5] → weights [0.78, 0.13, 0.09]
            # The most relevant past tokens get the highest weights.
            attn_weights = softmax(attn_logits)

            # Compute the weighted sum of value vectors.
            # For each dimension j of the output:
            #   head_out[j] = weight_0 * v_h[0][j] + weight_1 * v_h[1][j] + ...
            #
            # This is the core of attention: it creates a "blended summary" of
            # past tokens, where more relevant tokens contribute more.
            # e.g., if processing "Mich" and 'M' gets weight 0.6 while 'i' gets
            # 0.3 and 'c' gets 0.1, the output is mostly influenced by 'M'.
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]

            # Append this head's 4-dim output to the growing list.
            # After all 4 heads: x_attn has 16 values (4 heads × 4 dims).
            x_attn.extend(head_out)

        # Project the concatenated multi-head output back through a linear layer.
        # This lets the model learn how to combine information from different heads.
        # attn_wo mixes the outputs: maybe head 0's vowel pattern + head 2's
        # consonant pattern together produce a useful combined signal.
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])

        # RESIDUAL CONNECTION: add back the original input we saved earlier.
        # x = attention_output + original_input
        # This ensures the model can always "fall back" to what it had before,
        # and the attention only needs to learn what to ADD, not reconstruct
        # everything from scratch. Critical for training stability.
        x = [a + b for a, b in zip(x, x_residual)]

        # =====================================================================
        # 2) MLP (FEED-FORWARD) BLOCK
        # Purpose: after attention has gathered context from other tokens,
        # the MLP does the actual "thinking" — transforming the combined
        # information into a more useful representation.
        # =====================================================================

        # Save x again for another residual connection.
        x_residual = x
        # Normalize before the MLP.
        x = rmsnorm(x)
        # Expand: project from 16 dimensions → 64 dimensions.
        # This wider space gives the network more room to represent complex
        # patterns. Think of it as "brainstorming" with more dimensions.
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        # ReLU activation: max(0, x) for each element.
        # This introduces non-linearity — without it, two stacked linear layers
        # would just be equivalent to one linear layer (matrix × matrix = matrix).
        # ReLU lets the network learn curves, not just straight lines.
        # It also creates "sparsity" — roughly half the values become 0, so
        # only the most active "neurons" contribute to the output.
        x = [xi.relu() for xi in x]
        # Compress: project from 64 dimensions → 16 dimensions.
        # Squeeze the brainstormed ideas back into the standard embedding size.
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        # Residual connection again: add the MLP's contribution to what we had.
        x = [a + b for a, b in zip(x, x_residual)]

    # --- Step 3: Produce output logits ---
    # Project the final 16-dim representation to 27 scores (one per vocab token).
    # The lm_head matrix (27 × 16) computes a dot product of x with each token's
    # "ideal vector" — the token whose ideal vector is most similar to x gets
    # the highest score.
    logits = linear(x, state_dict['lm_head'])
    # Return raw scores. The caller will apply softmax to get probabilities.
    return logits

# --------------------------------------------------------------------------
# Adam Optimizer Setup
#
# The optimizer's job: after backward() computes how each parameter should
# change (the gradient), the optimizer decides HOW MUCH to actually change it.
# Adam is smarter than basic gradient descent — it adapts the step size for
# each parameter individually and uses momentum to smooth out noisy gradients.
# --------------------------------------------------------------------------

learning_rate = 0.01  # The base step size for parameter updates.
                      # Too large → the model overshoots and never converges
                      # (like taking huge leaps and jumping over the valley floor).
                      # Too small → learning is painfully slow.
                      # 0.01 is a common starting point for small models.
                      # (Note: this gets decayed linearly during training — see lr_t below.)

beta1 = 0.85  # Momentum decay rate (for the 1st moment / moving average of gradients).
              # Controls how much "memory" the optimizer has of past gradients.
              # beta1=0.85 means: new_momentum = 85% old_momentum + 15% current_gradient.
              # This smooths out noisy gradients — if the gradient flip-flops between
              # positive and negative, momentum averages it out so the parameter
              # doesn't jerk back and forth. Like a heavy ball rolling downhill —
              # it builds speed in a consistent direction and ignores small bumps.
              # Standard value is 0.9; 0.85 here means slightly less memory / more
              # responsive to recent gradients.

beta2 = 0.99  # Velocity decay rate (for the 2nd moment / moving average of squared gradients).
              # Tracks how LARGE the gradients have been recently (regardless of direction).
              # beta2=0.99 means: new_velocity = 99% old_velocity + 1% current_gradient².
              # This is used to ADAPT the step size per parameter:
              #   - Parameters with consistently large gradients get SMALLER steps
              #     (they're already changing a lot, so be cautious).
              #   - Parameters with tiny gradients get BIGGER steps
              #     (they need more help to learn, so push them harder).
              # 0.99 is the standard value — a long memory for gradient magnitudes.

eps_adam = 1e-8  # A tiny number (0.00000001) added to prevent division by zero.
                 # In the update formula, we divide by sqrt(v_hat). If v_hat
                 # happens to be exactly 0 (no gradients seen yet), we'd get
                 # a division-by-zero error. Adding epsilon avoids this.
                 # It's so small it doesn't affect the math in any meaningful way.

# Optimizer state buffers — one entry per parameter (~11,000 each):
m = [0.0] * len(params)  # First moment buffer (momentum): running average of gradients.
                          # m[i] tracks "which direction has parameter i been pushed recently?"
                          # Starts at 0.0 for every parameter (no history yet).

v = [0.0] * len(params)  # Second moment buffer (velocity): running average of squared gradients.
                          # v[i] tracks "how big have the gradients for parameter i been?"
                          # Starts at 0.0 for every parameter (no history yet).
                          # Note: this 'v' is the optimizer buffer, NOT the 'v' (values) in attention.

# ==========================================================================
# TRAINING LOOP
#
# This is where the model actually LEARNS. The loop repeats 1000 times, and
# each iteration follows the same 4-step recipe:
#   1. Pick a name and tokenize it (prepare the input)
#   2. Forward pass: run the model and measure how wrong it is (loss)
#   3. Backward pass: figure out which parameters caused the mistakes (gradients)
#   4. Update: nudge each parameter to make the loss a little smaller (Adam)
# ==========================================================================

num_steps = 1000 # Total number of training steps. More steps = more learning,
                 # but diminishing returns eventually. 1000 is enough for this
                 # tiny model to learn basic name patterns.

for step in range(num_steps):

    # ======================================================================
    # Step 1: PREPARE THE INPUT — pick a name, convert it to token IDs
    # ======================================================================

    # Pick one name from the dataset. The modulo (%) wraps around if we run
    # more steps than we have names, so we'd cycle through the dataset again.
    doc = docs[step % len(docs)]

    # Convert the name to a list of token IDs, surrounded by BOS on both sides.
    # Example: "Emma" → [BOS, E, m, m, a, BOS] → [26, 4, 12, 12, 0, 26]
    # The leading BOS tells the model "a name is starting."
    # The trailing BOS tells the model "the name has ended."
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]

    # n = number of (input, target) pairs we can make from this sequence.
    # For "Emma" with 6 tokens: n = min(16, 6-1) = 5 pairs:
    #   (BOS→E), (E→m), (m→m), (m→a), (a→BOS)
    # The min with block_size ensures we don't exceed the model's max length.
    n = min(block_size, len(tokens) - 1)

    # ======================================================================
    # Step 2: FORWARD PASS — run the model on each token, compute the loss
    # ======================================================================

    # Initialize fresh KV caches for this name. Each layer gets an empty list
    # that will accumulate keys and values as we process each token.
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []

    # Process each position in the name one at a time, left to right.
    for pos_id in range(n):
        # token_id:  the current input token  (what we feed IN)
        # target_id: the next token           (what we want the model to predict)
        # e.g., for "Emma" at pos_id=0: token_id=BOS, target_id=E
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]

        # Run the GPT model: given this token and its position, get 27 logits
        # (raw scores for what comes next). The KV cache accumulates so each
        # new token can attend to all previous tokens.
        logits = gpt(token_id, pos_id, keys, values)

        # Convert logits to probabilities via softmax.
        # e.g., [0.02, 0.01, ..., 0.40, ..., 0.03] — one probability per token.
        probs = softmax(logits)

        # CROSS-ENTROPY LOSS for this position:
        # Look up the probability the model assigned to the CORRECT next token,
        # then take -log of it.
        #
        # Why -log?
        #   - If prob = 1.0 (perfect prediction): -log(1.0) = 0    → zero loss, great!
        #   - If prob = 0.5 (coin flip):          -log(0.5) = 0.69 → moderate loss
        #   - If prob = 0.01 (nearly wrong):      -log(0.01) = 4.6 → huge loss, bad!
        #
        # The -log function heavily punishes confident wrong predictions and
        # barely rewards already-correct ones. This pushes the model to put
        # high probability on the right answer.
        loss_t = -probs[target_id].log()
        losses.append(loss_t)

    # Average the per-position losses into one scalar loss for this name.
    # We divide by n so that short names and long names contribute equally
    # (otherwise long names would dominate the gradient just by having more terms).
    # This single Value is the ROOT of the computation graph — every operation
    # that produced it (all the gpt() calls, softmax, log, etc.) is recorded
    # in the graph and can be backpropagated through.
    loss = (1 / n) * sum(losses)

    # ======================================================================
    # Step 3: BACKWARD PASS — compute gradients for every parameter
    # ======================================================================

    # Walk backward through the entire computation graph (from loss all the way
    # back to every weight in state_dict), computing how much each parameter
    # contributed to the loss. After this call, every Value in params has its
    # .grad field filled in with the derivative of the loss w.r.t. that param.
    loss.backward()

    # ======================================================================
    # Step 4: ADAM OPTIMIZER UPDATE — adjust parameters to reduce the loss
    # ======================================================================

    # Linear learning rate decay: start at 0.01 and linearly decrease to 0.
    # At step 0:    lr_t = 0.01 * (1 - 0/1000) = 0.01    (full speed)
    # At step 500:  lr_t = 0.01 * (1 - 500/1000) = 0.005 (half speed)
    # At step 999:  lr_t = 0.01 * (1 - 999/1000) ≈ 0.00001 (crawling)
    # This helps: big steps early to make fast progress, tiny steps later to
    # fine-tune without overshooting. Like sanding wood — coarse grit first,
    # then fine grit for a smooth finish.
    lr_t = learning_rate * (1 - step / num_steps)

    # Update every parameter using the Adam algorithm.
    for i, p in enumerate(params):
        # Update the first moment (momentum): exponential moving average of gradients.
        # m[i] remembers the "direction" this parameter has been pushed recently.
        # 85% of the old direction + 15% of today's gradient.
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad

        # Update the second moment (velocity): exponential moving average of SQUARED gradients.
        # v[i] remembers the "magnitude" of recent gradients (always positive).
        # 99% of the old magnitude + 1% of today's gradient².
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2

        # Bias correction for the first moment.
        # Since m starts at 0, early values are biased toward 0. Dividing by
        # (1 - beta1^t) corrects this. At step 1: divide by (1 - 0.85) = 0.15,
        # making m_hat ~6.7× larger than m. By step 20, the correction is tiny.
        m_hat = m[i] / (1 - beta1 ** (step + 1))

        # Bias correction for the second moment (same idea).
        # At step 1: divide by (1 - 0.99) = 0.01, a 100× boost.
        v_hat = v[i] / (1 - beta2 ** (step + 1))

        # THE ACTUAL UPDATE: nudge the parameter in the direction that reduces loss.
        #   p.data -= lr_t * m_hat / (sqrt(v_hat) + epsilon)
        #
        # Breaking it down:
        #   - m_hat: the smoothed gradient direction (which way to push)
        #   - sqrt(v_hat): the typical gradient size (how jumpy this param is)
        #   - m_hat / sqrt(v_hat): normalize the step — big gradients get smaller
        #     steps, small gradients get bigger steps. Each parameter gets a
        #     personalized step size.
        #   - lr_t: scale everything by the (decaying) learning rate
        #   - -= : subtract because we want to go DOWNHILL (reduce the loss).
        #     The gradient points uphill (direction of steepest increase), so
        #     we go the opposite way.
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)

        # Reset the gradient to 0 for the next training step.
        # If we didn't do this, gradients would accumulate across steps,
        # and the model would be updating based on stale information from
        # previous names.
        p.grad = 0

    # Print progress: the loss should generally decrease over time.
    # A loss of ~3.3 is random guessing (ln(27) ≈ 3.3 for 27 tokens).
    # A loss of ~2.0 means the model is learning patterns.
    # A loss of ~1.0 means it's getting quite good at predicting next letters.
    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

# ==========================================================================
# INFERENCE — Generate new names the model has never seen
#
# Training is done. The ~11,000 parameters are now tuned. Now we use the model
# to GENERATE brand new names by sampling one letter at a time. This is called
# "inference" or "generation."
#
# The process is autoregressive ("self-feeding"):
#   1. Start with BOS (beginning of sequence)
#   2. Ask the model: "given what you've seen so far, what comes next?"
#   3. Randomly pick a letter based on the model's probabilities
#   4. Feed that letter back in as input, repeat from step 2
#   5. Stop when the model outputs BOS (end of sequence)
#
# This is EXACTLY how ChatGPT generates text — one token at a time, each
# token conditioned on everything before it. The only difference is scale.
# ==========================================================================

# Temperature controls the "creativity" vs "predictability" tradeoff.
# It works by dividing the logits before softmax:
#
#   Low temperature (e.g., 0.1):
#     Logits [2.0, 1.0, 0.5] / 0.1 → [20, 10, 5] → softmax → [0.9999, 0.0001, 0.0000]
#     The distribution becomes very PEAKED — the model almost always picks the
#     highest-probability letter. Names will be safe and repetitive.
#
#   High temperature (e.g., 1.0):
#     Logits [2.0, 1.0, 0.5] / 1.0 → [2.0, 1.0, 0.5] → softmax → [0.51, 0.24, 0.11]
#     The distribution stays SPREAD OUT — lower-probability letters still have a
#     decent chance. Names will be more creative but sometimes nonsensical.
#
#   Temperature = 0.5 (used here): a middle ground — mostly picks likely letters
#   but occasionally takes creative detours.
temperature = 0.5

print("\n--- inference (new, hallucinated names) ---")

# Generate 20 new names.
for sample_idx in range(20):

    # Start with a fresh KV cache — each generated name is independent.
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]

    # Begin with the BOS token, just like during training.
    # This tells the model: "a new name is starting — what's the first letter?"
    token_id = BOS

    # Collect generated characters here.
    sample = []

    # Generate up to block_size (16) characters for this name.
    for pos_id in range(block_size):

        # Run the model: given the current token and position, get 27 logits.
        # The KV cache remembers all previous tokens, so the model has full
        # context of the name generated so far.
        logits = gpt(token_id, pos_id, keys, values)

        # Apply temperature scaling BEFORE softmax.
        # Dividing each logit by temperature sharpens (T<1) or flattens (T>1)
        # the probability distribution.
        # Then convert the scaled logits to probabilities.
        probs = softmax([l / temperature for l in logits])

        # SAMPLE a token randomly, weighted by the probabilities.
        # random.choices picks from range(27) where each token's chance of
        # being picked equals its probability. This is what makes generation
        # non-deterministic — run it twice, get different names!
        # [0] because random.choices returns a list; we just want the one token.
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]

        # If the model outputs BOS, it's saying "the name is done."
        # Stop generating and move on to the next name.
        if token_id == BOS:
            break

        # Otherwise, convert the token ID back to its character and save it.
        # e.g., token_id=0 → 'a', token_id=4 → 'e'
        sample.append(uchars[token_id])

    # Print the generated name by joining all the characters together.
    # e.g., ['M', 'a', 'r', 'i', 'a'] → "Maria"
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
