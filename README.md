# End-to-End Self-Driving CNN

A convolutional network that steers a car in the [Udacity self-driving simulator](https://github.com/udacity/self-driving-car-sim) from a single camera frame. It is trained with behavioral cloning (NVIDIA [PilotNet](https://arxiv.org/abs/1604.07316)) and completes a lap without leaving the track.


# Demo
[Watch the demo video here.](https://youtu.be/q___c4VQNL8) 

---
# Pipeline:
1. collect driving logs 
2. rebalance the steering distribution 
3. train with on-the-fly augmentation
4. run the saved model live.

The hard part was not the architecture. Raw logs are almost all “go straight,” so a model trained on them learns the laziest strategy: always predict steering ≈ 0. The rest of this README covers how that got fixed.
---

# EDA (`EDA.ipynb`)

The simulator writes `driving_log.csv` with 7 columns: 
- columns 1-3: paths to the center/left/right camera images, 
- column 4: steering (-1 to 1)
- column 5: throttle (0-1)
- column 5: brake (0-1)
- column 6: speed

The notebook does two things:

1. **Explore the data.** Histograms of steering and speed, plus left/right/straight counts. Findings: the steering distribution is a huge spike at zero (the track is mostly straight), turns are roughly balanced left vs right, and most frames are at full throttle.
2. **Balance the data.** Two versions exist side by side: the old manual deadzone approach, and the current Freedman–Diaconis binning (explained in the next section). The balanced result is saved as `balanced_log.csv`, which is what `train.ipynb` reads.

---
# Why the Freedman–Diaconis Rule?



## The problem it solves

The recorded steering values are overwhelmingly near zero due to the track being mostly straight; most frames are just "keep going straight." Training on that raw distribution produces a model that mostly predicts **steering ≈ 0.**

With mean squared error, guessing zero is already correct most of the time, so there is little pressure to learn corners. The symptom was a car that would barely turn on corners.

Fixing this means **reshaping the training distribution** so that turns are not drowned out by straights.

### The first attempt, and why it wasn't good enough

The original approach used a manual deadzone: treat `|steering| < 0.025` as “near straight,” keep 45% of those frames, and keep every turn.


It partially worked, but had 2 problems:

- **Two magic numbers.** `0.025` and `0.45` were picked arbitrarily.
- **Binary thinking.** A 0.03 wiggle and a 0.40 hard corner were both classified as a "a turn," despite being completely different driving events.



### Reframing the goal

The deadzone approach asks *"is this frame a turn?"*. The better question is *"is any part of the steering range over-represented?"*. Essentially, the goal is to **downsample the straights**.

Rather than manually deciding which steering values count as "straight," the steering range is divided into equal-width intervals (`bin_width`). The count in each interval then reveals which parts of the range are over-represented. In this dataset, those are the intervals near zero steering.

The Freedman–Diaconis rule is used only to choose a suitable width for these intervals. It estimates the width from the spread and size of the dataset instead of using a manually selected deadzone or number of bins:

```
bin_width = 2 × IQR / n^(1/3)
```



### Why IQR, and why the cube root

`IQR` **sets the scale.** Bin width has to be proportional to how spread out the data is. One way to measure spread is standard deviation using Scott's rule, `3.49 × σ × n^(-1/3)`. But standard deviation is dragged upward by sharp turns. This data is mouse-recorded and contains occasional sharp corrections out near the steering limits, and under Scott's rule those few frames would widen *every* bin in the histogram due to the sensitivity of standard deviation to outliers. 

`n^(1/3)` **links bin width to the amount of data.** Since `n` is in the denominator, more data = narrower bins. That makes sense because narrow bins only work when there is enough data:

- If bins are too **wide**, different steering behaviours get lumped into the same bar (a 0.03 wiggle and a 0.10 turn could share a bin).
- If bins are too **narrow**, each bin only holds a few frames, and the counts are basically random luck instead of a real pattern.

So there is a sweet spot: as narrow as possible, but each bin still needs enough frames in it to be worthwhile. How narrow you can go depends on how much data you have. The cube root is the rate (proven in the Freedman–Diaconis paper) at which you can safely shrink the bins as the dataset grows.

### Implementation:

1. Keep only full-throttle frames (`throttle == 1`) so speed is roughly constant.
2. Compute `bin_width` from the IQR and `n`, convert it to a bin count, and label every frame with `pd.cut`.
3. Cap each bin at the **mean bin count** (62 frames here) ; sample randomly inside over-full bins, keep under-full bins entirely.
4. Shuffle and write out `balanced_log.csv`.

This cut the dataset from **7,667 frames to 3,039**. The near-zero bins are the ones that got trimmed; the hard-corner bins were kept in full.

### What this does *not* solve

- **The per-bin cap is still arbitrary.** `max_per_bin = counts.mean()` is a sensible target, but nothing derives it.

### References:
[Freedman–Diaconis rule - Wikipedia](https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule)


[Freedman, D. and Diaconis, P. (1981) On the Histogram as a Density Estimator: L2 Theory.](https://doi.org/10.1007/BF01025868)


---

# Training (`train.ipynb`)

1. **Load and split:** Reads the balanced log, then an 80/20 train/validation split (`random_state=42` so the split is reproducible).
2. **Test Cells:** One-image test cells for `random_augment` and `preprocess`, so I can see what the model actually receives before training on it.
3. **Filter missing images:** Some rows in the log point to image files that no longer exist on disk (deleted/corrupted/missing). Dropping those rows up front stops the generator from producing short or empty batches mid-training.

## DataGenerator

Builds batches on the fly instead of precomputing X/y once. This is what makes augmentation **dynamic**: every epoch re-rolls the random choices, so the model has a lower chance of seeing the same batch in the next epoch. Per training sample:

- Randomly pick center/left/right camera. Side cameras get a **±0.2 steering correction** — a left-camera frame looks like the car drifted left, so the label is adjusted toward steering back. This is free "recovery" data.
- Apply `random_augment` (flip/brightness).
- Apply `preprocess` so training images match what the simulator will feed the model.

**Validation batches skip all of it (center camera, no augments) so `val_loss` measures the real task.**

   Developed as a group course project. Each member built an independent pipeline; this repo is my implementation. The on-the-fly DataGenerator approach was inspired by a teammate’s code. 
## Model

Layers follow [Bojarski et al. 2016](https://arxiv.org/abs/1604.07316):

 5 conv layers, then 3 dense layers down to a **single linear output** ; this is regression (continious value prediction), not classification so there's no softmax.

Training setup:

- **Adam @ 1e-4, MSE loss**: standard for regression.
- **EarlyStopping** (patience 5 on `val_loss`): stops when validation stops improving, restores the best weights.
- **ModelCheckpoint**: saves the best `val_loss` model to `model.h5` during training, so a bad final epoch can't overwrite a good model.
- Loss curves are plotted at the end to check for overfitting (train loss dropping while val loss rises).

---

# Support scripts

**`preprocess.py`**: the pipeline every image goes through, in training *and* in the simulator: crop to the road region → RGB-to-YUV → Gaussian blur → resize to 200×66 → normalize to [0, 1]. It must be identical in both places, otherwise the model gets inputs at test time that it never saw in training.

**`augmentations.py`**: `random_augment` applies flip (mirrors the image **and negates the steering label**) and random brightness (steering unchanged). **Zoom, pan, and rotate exist but are disabled**: they change what the correct steering *should be* without updating the label, which teaches the model wrong answers.

---

# Running the simulator

Use **Python 3.10 or 3.11** (not 3.12/3.13). Newer Python breaks `eventlet` (`ssl.wrap_socket` was removed), and the Udacity bridge needs older `socketio` / `engineio` / `eventlet` packages.

1. Download the [Udacity simulator](https://github.com/udacity/self-driving-car-sim). Make sure you download the **Term 1** release:
      - [Udacity simulator](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-windows.zip)


2. Create and activate a virtualenv in the project folder:
   ```bash
   py -3.11 -m pip install virtualenv
   py -3.11 -m virtualenv .venv
   .venv\Scripts\activate
   ```
3. Install the bridge + ML deps (pin these versions so they match the simulator):
   ```bash
   pip install tensorflow keras opencv-python pillow flask
   pip install "python-socketio==4.6.1" "python-engineio==3.14.2" "eventlet==0.30.2"
   ```

4. Make sure `model_best_working.h5` (or whatever `TestSimulation.py` loads) is in the project root.

5. Start the server:
   ```bash
   python TestSimulation.py
   ```
   It listens on port **4567**. When it prints something like “waiting” / starts the WSGI server, it’s ready — not stuck.

6. Open the Udacity simulator → **Autonomous Mode** → select the track. The car should connect (`Connected` in the terminal) and start receiving steering/throttle commands.

Notes:
- Training and simulation must use the same preprocess (`preprocess.py`). `TestSimulation.py` already calls it.
- Current script uses `maxSpeed = 30` and multiplies predicted steering by `1.5`.

---
# What did not work:

1. I took a racing line and found that the model was terrible because of it.
   - essentially, hitting the apex of corners, hugging the inside of a corner, going from outside to inside
      - this resulted in the car trying to start the turn from the outside, but failed to give enough steering input to corner, resulting in understeer.
2. Additionally, I used keyboard to generate the training data at first. The caveats for this include:
   - Speed affects turn radius; I gathered data at a consistent 30km/h with the throttle down all the way.
      - At a lower speed, the steering inputs ***may*** have result in understeer (less distance is covered at 'x' steering, before the steering value goes back to 0)
      - I even tried to multiply the predicted steering by 1.5 but that didn't fix it.
3. I realized that I was applying Augments statically (if you augment once when building X / y, every epoch sees the same images). I needed to use DataGenerator to apply random augments for each epoch.