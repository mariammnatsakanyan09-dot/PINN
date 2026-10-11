# Task 5 – PINN / neural ODE for the Silverbox oscillator

## Run it (needs the real data file)

1. Download `SNLS80mV.mat` from https://www.nonlinearbenchmark.org/ and put it next to the script.
2. `pip install torch scipy numpy matplotlib h5py`
3. Train all settings (about 10 min per setting on one CPU core; split across terminals with `--only`):

```
python pinn_silverbox.py run --data SNLS80mV.mat --out runs
# or, in two terminals:
python pinn_silverbox.py run --data SNLS80mV.mat --out runs --only w32x2_tanh_adam,w64x2_tanh_adam,w64x3_tanh_adam,w128x2_tanh_adam
python pinn_silverbox.py run --data SNLS80mV.mat --out runs --only w64x2_silu_adam,w64x2_tanh_adamw,w64x2_tanh_adam_short,w64x2_tanh_adam_long
```
4. `python pinn_silverbox.py summary --out runs` writes `runs/summary.md` (table of all settings with
   train/test RMSE, best one marked), and `runs/best_test_plot.png` (measured vs predicted on the test segment).

The edit-able sweep list is `CONFIGS` in the script.

## What the code does

* **Data.** Slice `[42650:46746]` of V1 (input u) and V2 (output y); subtract the mean of u and the mean of y
  over that slice. First 3072 samples = training record, last 1024 = test record. `dt = 1/610.35`.
* **State equation.** `y' = v`, `v' = alpha_theta(y, v, u)`. `alpha_theta` is a fully connected PyTorch
  network (3 inputs -> hidden layers -> 1 output) that has to represent the whole force
  `g u - a v - b y - c y^3`; the coefficients a, b, c, g are not given to the network.
  Inputs are divided by their training-record standard deviations (y, v, u) and the output is multiplied by the
  standard deviation of the finite-difference acceleration, so the network works with O(1) numbers.
* **Integrator.** Classical RK4 with step `dt`, driven by the measured input (`u` at the half step is the average of
  the two neighbouring samples).
* **Training (training record only).**
  1. *Warm start (physics-informed collocation):* fit `alpha_theta(y_k, v_k, u_k)` to the finite-difference
     acceleration at the training samples.
  2. *Multiple shooting:* random windows of length L are integrated with RK4 from the measured position and
     finite-difference velocity; loss = MSE between simulated and measured y. L grows in stages
     (16 -> 64 -> 256 steps), learning rate falls with a cosine schedule inside each stage, gradient clipping 1.0.
     Training windows never reach beyond sample 3071.
* **Metric.** One free run of the whole 4096-sample excerpt: start at sample 0 with measured `y[0]` and a
  finite-difference velocity (second-order one-sided formula), integrate through the training samples and on
  into the test samples with the same state, reading only `u`. RMSE on the last 1024 samples is the test score;
  the same run restricted to the first 3072 samples is the train score. The zero predictor scores the RMS of the
  test output (about 0.06 V).



