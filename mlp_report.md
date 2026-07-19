# MLP Development Report

This report walks through how I built the Multi-Layer Perceptron, the bugs
I ran into while building it, and the experiments I ran on real images to
understand its behaviour.

## Starting point

I started by splitting the MLP across four files. The main `mod.rs` holds
the `MLP` struct itself along with `forward`, `backward`, `train`,
`predict`, and `accuracy`. A `layer.rs` file defines a Dense `Layer` with
its weights, biases, and activation function. An `activation.rs` file
contains the activation functions and their derivatives (Tanh, ReLU). And
finally `initializer.rs` provides He initialisation for the weights.

The mathematical foundation comes straight from the course slides. The
output layer delta is computed as `activation_derivative(x) × (x − y)`,
the hidden layer delta as `activation_derivative(x) × Σ(w × δ)`, and the
weights are updated by `w ← w − α × x × δ`. Targets are built with `+1`
for the correct class and `−1` for all others, since Tanh outputs land in
that range.

## Some design choices

A few things mattered enough to me that I want to mention them.

The activation function is stored on each layer as a function pointer, not
hardcoded. That way the same MLP works with Tanh, ReLU, or anything else
the caller wants. The output layer's activation is also a separate
parameter, distinct from the hidden layers, so the same code can do
classification (Tanh output) or regression (identity output) without any
branching.

The architecture itself is configurable through a slice of layer sizes
passed to `MLP::new`. I can pass `&[20, 16, 3]` for a small classifier or
`&[3072, 64, 32, 16, 3]` for something deeper, and the constructor just
builds the right number of layers automatically.

## First bug: the network refused to learn

After everything compiled cleanly, I ran my first training and got
accuracy stuck at exactly 33.3% across 5000 epochs. For a 3-class problem
that's chance level. The loss barely moved. The network was effectively
frozen.

The bug was in `he_init`. I was multiplying a tiny random range by the He
scaling factor, like this:

```rust
let value = rng.random_range(-0.01..0.01) * (2.0 / n_inputs as f32).sqrt();
```

With weights initialised around ±0.01, every neuron output was essentially
zero, every class score was nearly identical, and gradients vanished
before they could meaningfully update anything. The fix was to use the
proper random range and let the He factor do its job:

```rust
let value = rng.random_range(-1.0..1.0) * (2.0 / n_inputs as f32).sqrt();
```

After that, training worked immediately. On synthetic test points the MLP
hit 100% accuracy quickly.

## Second bug: shape mismatch in backprop

The next problem appeared once I started training on real data. The
program crashed with a shape mismatch in matrix subtraction, complaining
about shapes 3 and 8 not matching.

The cause was the order of operations inside my backward loop. I was
computing the delta for the previous layer *before* updating the current
layer's weights. So by the time the code tried to use `delta` for the
current layer's gradient, it had already been overwritten with the
previous layer's shape.

```rust
// what I had — broken
for l in (0..self.layers.len()).rev() {
    delta = ...;                          // wrong shape for THIS layer now
    self.layers[l].weights = ...;         // crashes here
}
```

The fix was simply to swap the two steps. Update the current layer first,
then transform the delta into the previous layer's shape for the next
iteration:

```rust
for l in (0..self.layers.len()).rev() {
    self.layers[l].weights = ...;        // use delta while it still matches
    if l > 0 {
        delta = ...;                      // now transform it for next time
    }
}
```

This matches the standard backprop ordering, which I had read about but
not internalised until the crash forced me to think about it carefully.

## Training on real images

With the MLP working correctly on synthetic data, I moved on to actual
cat, lion, and cheetah photos. The first run was on a small subset of the
images, using raw pixel values as the input features. Each image was
resized to 32×32 RGB, which produces 3072 features per image.

With 600 images total and an architecture of `[3072, 16, 3]`, the model
has roughly 49,200 parameters.

The results were:

| Set | Accuracy |
|-----|----------|
| Train | 75.0% |
| Test  | 42.5% |

That 32.5 point gap is severe overfitting. The model wasn't generalising
at all. It was memorising training images and failing on anything new.

This made sense in hindsight. The course slides give a useful rule of
thumb: training examples should be at least ten times the number of
parameters. With 480 training samples and 49,200 parameters, I was over a
hundred times below that recommendation.

## Trying more data

The obvious next move was to load more images. I bumped the sample count
to 3000 per class (9000 total), giving 7200 training samples.

The new results:

| Set | Accuracy |
|-----|----------|
| Train | 47.7% |
| Test  | 47.5% |

The overfitting was gone (train and test were nearly identical), but
accuracy was now barely above the 33% random baseline. The model had
swung from one extreme to the other. With 50,000 parameters trying to
learn anything meaningful using only stochastic updates, training was too
slow and converged on a weak solution.

This is the bias-variance tradeoff in practice. The first experiment had
too much variance (memorising noise). The second had too much bias (not
expressive enough relative to the data). I needed something in the
middle.

## Engineered features

Raw pixels are a poor representation for a small classical model. The
information that distinguishes the three animals — coat colour, spots
versus solid, texture — is scattered across thousands of correlated pixel
values. Instead of asking the model to discover all of that from scratch
with limited data, I extracted 20 meaningful features per image:

Six features for colour statistics (mean and standard deviation per RGB
channel). Eight features for a brightness histogram in normalised buckets.
Six features for edge density (three channels at two thresholds).

With 20 features and `[20, 16, 3]` architecture, the model now has around
400 parameters. With 7200 training samples that puts me well within the
parameter budget rule, even with margin to spare.

The result on the same data, same architecture, only the feature
extractor changed:

| Set | Accuracy |
|-----|----------|
| Train | 57.5% |
| Test  | 55.8% |

Train and test track together. Both improved substantially over the
raw-pixel run. The gap is under two points, which is what a healthy model
looks like.

## Pushing capacity

To see what would happen if I gave the model more room, I tried a deeper
architecture `[20, 32, 16, 3]`:

| Set | Accuracy |
|-----|----------|
| Train | 60.6% |
| Test  | 55.7% |

Training accuracy climbed three points but test accuracy stayed flat. The
extra capacity went into overfitting, not learning. So `[20, 16, 3]` is
close to the sweet spot for this feature set.

## Looking at where the model is failing

The 55% headline number hides a lot. I built a confusion matrix on the
test set to see which classes were causing trouble:

```
                 pred cat   pred cheetah   pred lion
   true cat          497            58          44
   true cheetah      128           340          76
   true lion         194           302         161
```

Computing per-class accuracy from this:

| Class | Correct | Per-class accuracy |
|-------|---------|--------------------|
| Cat | 497 / 599 | 82.9% |
| Cheetah | 340 / 544 | 62.5% |
| Lion | 161 / 657 | 24.5% |

Cats are easy. Cheetahs are moderate. Lions are essentially being guessed
at, which is even worse than random for a 3-class problem.

The interesting pattern is in the lion row. The model frequently predicts
cheetah for actual lions (302 out of 657 lions). That makes sense if you
think about it. My feature set captures colour and texture, but both
animals are tawny, and a lion's distinguishing characteristics (mane, body
shape, posture) are not in my descriptors at all. The features simply
can't tell the two apart.

This is not a model bug. It is a ceiling imposed by the choice of
features. To break past it I would need either spatial features like HOG,
or a fundamentally different model that handles the original pixels
better.

## Summary

| Stage | Configuration | Train | Test | Gap | Diagnosis |
|-------|---------------|-------|------|-----|-----------|
| 1 | Raw pixels, 600 samples | 75% | 43% | 32 | Overfitting |
| 2 | Raw pixels, 9000 samples | 48% | 48% | 0 | Underfitting |
| 3 | Engineered features, small MLP | 58% | 56% | 2 | Balanced |
| 4 | Engineered features, bigger MLP | 61% | 56% | 5 | Diminishing returns |

The journey through these four stages turned out to be the most
educational part of building the MLP. The two bugs were specific
mistakes, but the four-stage progression is a real demonstration of the
bias-variance tradeoff playing out on actual data, with the parameter
budget rule guiding each step.

The final 55% is not the model's hard limit. It is the limit of what 20
hand-picked descriptors can express about the difference between these
three animals. That distinction matters: the model architecture is fine,
the training works, the issue is what we're feeding it.
