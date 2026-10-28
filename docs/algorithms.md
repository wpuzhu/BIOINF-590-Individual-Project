# Algorithms and experimental protocol

## Model definitions

For dimensions $(d_0,\ldots,d_L)$, MLP parameters total $\sum_l(d_l+1)d_{l+1}$. KAN parameters total $\sum_l[(R+2)d_ld_{l+1}+d_{l+1}]$, counting R coefficients, a SiLU scale and a basis scale per edge, plus one bias per output node. There are no unused padding parameters. All basis families use R=8 by default. The scale/coefficient product is redundant under rescaling, as in many such parameterizations; equal raw scalar counts do not imply equal identifiable dimension.

The spline kernel is $B_3(z)=2/3-|z|^2+|z|^3/2$ for $|z|<1$, $(2-|z|)^3/6$ for $1\le|z|<2$, and zero otherwise. Centers are evenly spaced over [-1,1], with spacing h=2/(R-1). This is a fixed finite set of cardinal B-splines, without endpoint clamping, knot adaptation, or partition-of-unity guarantees at its finite boundary. RBFs use the same centers and width h. Fourier uses four sine/cosine pairs. No hidden normalization or input clipping is applied.

MLP hidden activations are tanh, ReLU or SiLU; its output is linear. KAN nodes sum nonlinear incoming edges at every layer, including the output layer. Regression uses squared error; classification uses cross-entropy on logits. No output softmax is used inside cross-entropy.

## Capacity-controlled depth comparison

For each dataset, seed, hidden depth and budget:

1. Search integer hidden widths for the closest active parameter count. For depths >1, preceding layers share width w and the final hidden width is between w/2 and 2w.
2. Reject the cell if the budget error exceeds 5%. Report widths, actual parameters and relative error.
3. Compare all six models at that depth and budget. Repeat for other depths without choosing depth from test performance.

Counts are close, not universally exactly equal. The biological budget is 4,096 because the increments possible in shallow 30-input KANs make a 1,024 budget poorly matched. Larger budgets and depths are provided in full configs. Capacity includes connectivity, basis placement and nonlinearity; this is not an isolation of architecture from learned activation flexibility.

## Training and validation-only selection

```
for each shared data split:
    for each architecture and parameter budget:
        for lr in the same candidate set:
            initialize deterministically
            repeat full-batch Adam updates:
                evaluate validation loss every 10 epochs
                halve lr after 4 stalled validation checks
                save every new best validation checkpoint
                stop after 12 checks without a >=0.1% relative improvement
            restore best checkpoint
        select learning rate by best validation loss
        evaluate selected model on each test split once
        record all candidate histories and summed search cost
```

Adam weight decay is 1e-4; gradient norm clipping is 10. All models get the same candidate LRs, stopping policy, full-batch observations and maximum allowance. The best checkpoint tracks any improvement; patience resets only after a meaningful relative improvement. Full-batch runtime includes forward/backward updates, validation checks and checkpoint copies, but excludes architecture search, initialization, data loading and final test evaluation. Search seconds sums every candidate's fit time. Prediction time is the median of 10 warmed batch forward passes divided by batch size; it is batch throughput per sample, not single-query latency. All recorded runs use CPU and one Torch thread. Do not compare these timings directly to JavaScript/browser runtimes or optimized external KAN libraries.

## Data protocol

Simulation: 1,000 train, 400 validation, 1,000 inside test and 1,000 outside test observations. Train/validation/inside inputs are uniform on [-1,1]^2. Outside inputs are uniform on [-1.5,1.5]^2 excluding [-1,1]^2. Noise SD=0.05; report noisy and noise-free RMSE. Independent RNG streams keep validation/test fixed when n_train changes. All models share identical splits within a seed.

Biology: stratified 60/20/20 train/validation/test. Fit StandardScaler only to training features. The test set is an IID held-out set; there is no biological extrapolation claim. Yeast's very rare classes make macro-F1 and per-split variation important. No class is dropped to improve results. Homology-aware protein splits and independent patient cohorts are future validation, not implemented guarantees.

## Reporting

Group results by dataset, depth, parameter budget and test region; report mean, SD and number of seeds. Use RMSE for regression and macro-F1, balanced accuracy, accuracy and log loss for classification. Save selected LR, epochs, best epoch, stop reason, fit seconds, search seconds and prediction throughput. Plot validation loss against elapsed time. Paired differences compare each KAN with MLP-SiLU on the same seeds; bootstrap intervals with few seeds are descriptive only. An epoch limit must remain visible rather than being called convergence.

The supplied full configs extend to 3 hidden layers, 2 budgets, 5 seeds and a 5,000-epoch cap. Even these settings may require more tuning or larger limits. Pilot results are an execution check and preliminary comparison, not a final claim of superiority.
