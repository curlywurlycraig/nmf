[Non-negative matrix factorisation](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization) is a method for factoring an input matrix into two other matrices. This has many applications, particularly in problems of determining how some feature vectors can be composed of other "template" feature vectors.

This Rust library currently implements a fast update step for the activation matrix (usually labelled "H") given some input ("V") and template library ("W"). This largely builds on the work in "Real-time polyphonic music transcription with non-negative matrix factorization and beta-divergence", by Arnaud Dessein et. al.

In the future, it may add support for update steps for "W" too.

# Usage

```rust
let mut nmf = FixedTemplateNmf::new(templates, activation_coef, &input, 0.5);

for _ in 1..5 {
    nmf.update_activation_coef();
}

let activation = nmf.get_activation_coef();
```
