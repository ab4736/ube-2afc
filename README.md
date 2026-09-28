# ube-2afc: pairmate discrimination from fMRI

Can you tell, from someone's brain response alone, which of two near-identical images they were
looking at? This runs that test on your own data, offline or live during a scan.

Built on the Universal Brain Encoder (UBE), [Beliy et al.](https://arxiv.org/abs/2406.12179).

## How it works

Two pairmate images are almost the same picture, so the usual approach of decoding the brain data
into CLIP features and asking which image it matches does not work. Both images have nearly the
same CLIP vector, so the information is not there to decode.

So we go the other direction. The encoder takes an image and predicts the brain pattern it should
produce. Predict a pattern for each of the two candidate images, then see which prediction the
measured pattern actually looks like:

```
pred_A = encoder(image A)
pred_B = encoder(image B)
correct if corr(measured beta, pred_A) > corr(measured beta, pred_B)
```

Chance is 0.50. Nothing is ever reconstructed, which is why this works where decoding fails.

## What this adds on top of UBE

The encoder architecture and the base pretraining are from the Irani lab. This repo adds:

- **The pairmate measurement.** Forced choice, CPD, retrieval, bootstrap confidence intervals, and
  leak controls. Their repo evaluates reconstruction quality and encoding accuracy, not this.
- **CPD during a scan.** Predictions are cached ahead of time so scoring a trial is two
  correlations and a projection, fast enough for a neurofeedback loop.
- **Fitting a new subject quickly.** Freeze the base, train only the per-voxel embeddings, about 20
  minutes on one GPU with a couple thousand images. Their code assumes NSD-scale data.
- **A pretraining objective that helps.** Adding a contrastive (InfoNCE + entropy) term to the base
  won on 8 of 8 NSD subjects (2AFC 0.870 to 0.954). Both bases are provided so you can compare.

Image reconstruction is not in here. If you want to reconstruct images from brain activity, use the
Brain-IT decoder in their repo.

## Installation instructions

1. Download this repository:

```bash
git clone https://github.com/ab4736/ube-2afc.git
cd ube-2afc
```

2. Create an environment with the packages needed to run the scripts:

```bash
conda create -n ube python=3.11 -y
conda activate ube
pip install -r requirements.txt
```

3. Open `config.sh` and fill in the top: the path to that Python, and your Slurm account and
partition if your cluster needs them. Nothing else in the repo holds a path, so this is the only
file you edit.

4. Download the pretrained base weights (about 10 MB each):

```bash
./download_checkpoints.sh
```

5. Cache the DINOv2 backbone. Run this on a **login node**, because compute nodes usually have no
internet:

```bash
./get_dinov2.sh
```

You need a Slurm cluster with one GPU, 16 GB or more.

## General information

This repository contains scripts for

- Building an input file from your data (`make_inputs.py`)
- Checking that input file before you waste time on it (`check_inputs.py`)
- Fitting the encoder to your subject (`train_encoder.py`)
- Scoring pairmate 2AFC and CPD (`eval_2afc.py`)
- Caching predictions so trials can be scored instantly (`make_pred_cache.py`)
- Scoring trials one at a time, after a session or during one (`score_trials.py`, `ube/online.py`)

`./submit.sh NAME` runs the middle four in order as one Slurm job, which is what most people want.

`extras/` has base pretraining from scratch, plus written-down notes on what worked and what did
not, so nobody repeats the dead ends.

## Pretrained models

You can skip pretraining entirely. `download_checkpoints.sh` fetches these:

- `ube_base_infonce.pt`: pretrained with the contrastive objective. **This is the default and the
  one to use.**
- `ube_base_recon.pt`: same architecture and data, reconstruction loss only. The matched baseline
  if you want to compare objectives.
- `ube_base_original.pt`: the earlier base this work started from.

Switch between them with the `BASE=` line in `config.sh`. They are small because the DINOv2
backbone is not inside them; it is stock and gets loaded separately.

## Preparing your data

You need three files plus your images.

**The betas.** One row per trial, one column per voxel. Apply your mask first so only the voxels
you want are in there. Single-trial betas are fine, repeats get averaged for you. Save from Python
with `np.save("betas.npy", betas)`, or from MATLAB with `save('betas.mat','betas','-v7')` with only
that one variable in the file.

**`trials.csv`.** One line per row of the betas, in the same order, saying which image was on
screen. Needs an `image` column. A `session` column is optional but recommended, since then each
voxel is z-scored within session.

```
image,session
scene_1023.png,1
pair_3_1.jpg,1
scene_88.jpg,1
scene_1023.png,2
pair_3_2.jpg,2
```

**`pairs.csv`.** One line per pairmate pair. Names must match `trials.csv` exactly.

```
image_a,image_b
pair_3_1.jpg,pair_3_2.jpg
pair_4_1.jpg,pair_4_2.jpg
```

The pairmate images become the test set and are taken out of training automatically, so the encoder
never sees them.

**The images.** Normal .jpg or .png, any size. They get resized to 224x224, and stretched if not
square, which is fine.

Every image that is not a pairmate gets used for training, so more is better. Around 2,000 works
well. Below roughly 1,000 it still runs but fits worse.

Then build the input file. Set `--name` to whatever you want to call this dataset, usually a
subject ID. Leave out `--images` if `trials.csv` already has full paths.

```bash
. ./config.sh
$PY make_inputs.py --name sub-08 --betas betas.npy --trials trials.csv \
    --pairs pairs.csv --images /path/to/images
```

```
$ python make_inputs.py --help
usage: make_inputs.py [-h] --name NAME --betas BETAS --trials TRIALS --pairs PAIRS
                      [--images IMAGES] [--val_frac VAL_FRAC] [--seed SEED]

options:
  --name NAME          what to call this dataset, e.g. sub-08
  --betas BETAS        .npy or .mat, one row per trial, one column per voxel
  --trials TRIALS      csv with an image column, one line per row of the betas
  --pairs PAIRS        csv with columns image_a,image_b
  --images IMAGES      folder with the images (skip if the csv has full paths)
  --val_frac VAL_FRAC  fraction held out for the health check
  --seed SEED          random seed for the train/val split
```

Then check it. This takes seconds and catches the mistakes that otherwise cost you a queue wait:

```bash
$PY check_inputs.py data/sub-08.npz
```

It prints how many images and voxels it found. `ALL GOOD` on the last line means you are fine. A
`PROBLEM:` line says exactly what to fix. `warning:` lines still run.

## Running the whole thing

```bash
./submit.sh sub-08
```

That queues one job which checks the inputs, fits the encoder (about 20 min), scores 2AFC and CPD,
and builds the prediction cache for online use. Watch it with `squeue -u $USER` and
`tail -f logs/ube_2afc_*.log`.

The rest of this section is what that job runs, if you want to do the steps yourself.

## Fitting the encoder to your subject

The base stays frozen. The only thing trained is one embedding vector per voxel of your subject,
which is why this is fast and works with a couple thousand images.

```bash
$PY train_encoder.py --data data/sub-08.npz --tag sub-08_infonce \
    --base checkpoints/ube_base_infonce.pt --lam 1 --went 0.1
```

- Set `--lam 1 --went 0.1` for the contrastive objective. This is the one that helps pairmates.
- Set `--lam 0 --went 0` for a plain reconstruction encoder, if you want the comparison.
- The saved file is a few MB, not 1.2 GB, because it stores the voxel embeddings and a pointer to
  which base they belong to.
- Runs are reproducible: the same `--seed` on the same input file gives the same numbers. Our
  reference subject comes out at 0.923 every time.

```
$ python train_encoder.py --help
usage: train_encoder.py [-h] --data DATA --tag TAG --base BASE [--lam LAM] [--tau TAU]
                        [--went WENT] [--epochs EPOCHS] [--bs BS] [--lr LR] [--seed SEED]
                        [--hub HUB] [--outdir OUTDIR] [--warm_embed WARM_EMBED]
                        [--warm_map WARM_MAP]

options:
  --data DATA           input npz from make_inputs.py
  --tag TAG             name for the output checkpoint
  --base BASE           pretrained base weights, e.g. checkpoints/ube_base_infonce.pt
  --lam LAM             weight on the contrastive term, 0 turns it off
  --tau TAU             contrastive temperature
  --went WENT           weight on the entropy term
  --epochs EPOCHS       passes over the training images
  --bs BS               batch size, also how many negatives the contrastive term sees
  --lr LR               learning rate
  --seed SEED           random seed
  --hub HUB             torch hub cache with dinov2, so it works offline
  --outdir OUTDIR       where to save the fitted encoder
  --warm_embed WARM_EMBED   optional nsd_voxel_embed_*.pt to warm start from
  --warm_map WARM_MAP   int .npy, one entry per voxel, which nsd voxel it matches
```

## Scoring 2AFC and CPD

```bash
$PY eval_2afc.py --data data/sub-08.npz --enc checkpoints/sub-08_infonce.pth
```

It prints:

```
  pairmate 2AFC   0.923  (24/26)  95% CI [0.81, 1.00]  chance 0.5
  CPD             mean +0.507  median +0.498  CPD>0 0.923
  retrieval       top-1 0.808  (1 of 26, chance 0.038)
  val retrieval   top-1 0.386  (1 of 233, chance 0.004)
  controls        shuffled_voxels 0.58  mean_beta 0.50  wrong_image 0.15
```

- **pairmate 2AFC** is the main number. Fraction of trials where the measured pattern matched the
  shown image's prediction better than its pairmate's. Each pair gives two trials, once with each
  image shown. The confidence interval resamples pairs.
- **CPD** projects the measured pattern onto the line between the two predictions. +1 means it sits
  right on the shown image's prediction, 0 is halfway, -1 is on the pairmate's.
- **retrieval** is a harder version: each measured pattern against the predictions for all the test
  images, not just its pairmate.
- **val retrieval** is the same thing on ordinary non-pairmate images, and it is the health check.
  If this is near chance the encoder did not fit your subject, and the 2AFC number means nothing.
- **controls** should all land near 0.50 (`wrong_image` can go below). `shuffled_voxels` scrambles
  the predictions, `mean_beta` uses the average pattern instead of the real one, and `wrong_image`
  uses some other scene's prediction. If any of these come out high, something is leaking and the
  result is not real.

You also get `results/sub-08_infonce_trials.csv`, one row per trial with both correlations, whether
it was correct, and the CPD. It opens in Excel.

## Scoring trials as they happen (neurofeedback)

Two parts. The first is useful on its own if you just want per-trial numbers after a session.

**Before the scan, cache the predictions.** `submit.sh` already did this. It runs the encoder once
per candidate image and saves the results, so at trial time there is no GPU work left.

```bash
$PY make_pred_cache.py --data data/sub-08.npz --enc checkpoints/sub-08_infonce.pth \
    --out cache/sub-08_preds.npz
```

**Score trials from a finished session.** `shown_foil.csv` has columns `shown,foil`, one line per
trial, matching the rows of your betas. Add `--causal_z` to normalise using only the trials seen so
far, which is the honest thing to do if you are imitating online conditions.

```bash
$PY score_trials.py --cache cache/sub-08_preds.npz --betas trial_betas.npy \
    --trials shown_foil.csv --out results/pertrial.csv
```

**Score trials live.** Use the scorer directly in your own scanner loop. It is pure numpy, no GPU,
so it takes microseconds and is never the bottleneck:

```python
from ube.online import PairmateScorer
scorer = PairmateScorer.from_cache("cache/sub-08_preds.npz")

scorer.update_running_stats(beta)     # feed each trial's beta in as it arrives
b = scorer.causal_z(beta)             # normalise using only the past
out = scorer.score(b, shown="pair_3_1.jpg", foil="pair_3_2.jpg", z=False)
print(out["cpd"], out["correct"])     # send out["cpd"] back as feedback
```

`realtime/rtcloud_example.py` is a filled-in template of that loop. The one part you supply is
turning the volumes collected so far into one beta for the current trial, because every real time
setup does that differently. A proper causal GLM works clearly better than averaging the volumes in
the window; on our data averaging came out near chance.

Two things we measured that matter for timing:

- Score at about 6 to 9 s after onset. Accuracy peaked at 9 s and waiting longer made it worse.
- Do not widen the window to try to decode earlier. A wide window has to include pre-response
  volumes, which delays it.

One caveat on CPD online: 2AFC does not care how your betas are scaled, but CPD does. If you change
how you normalise, CPD values shift, so only compare CPD computed the same way.

## Warm starting (optional)

By default a new subject's voxel embeddings start random. You can instead start each voxel from the
most similar NSD voxel, which helped on one of our subjects:

```bash
WITH_WARMSTART=1 ./download_checkpoints.sh      # adds a 324 MB file
$PY train_encoder.py --data data/sub-08.npz --tag sub-08_warm \
    --base checkpoints/ube_base_infonce.pt \
    --warm_embed checkpoints/nsd_voxel_embed_infonce.pt --warm_map my_voxel_to_nsd.npy
```

You have to build `my_voxel_to_nsd.npy` yourself: an integer array, one entry per voxel of your
subject, giving the index of the NSD voxel it corresponds to. That means putting your voxels and
the NSD voxels in a common space (fsaverage) and taking nearest neighbours. If you do not have
that, skip this. Random init works fine.

## Pretraining the base yourself

You do not need this. It takes the full NSD dataset and days of GPU time. If you want it,
`extras/train_base.py` does it, with `--objective recon` or `--objective infonce` selecting the two
objectives. `extras/README.md` explains what it needs and a batching detail that matters for the
contrastive term.

## The input file format

If you would rather build `data/NAME.npz` yourself instead of using `make_inputs.py`, it is a numpy
`.npz` with these arrays:

| key | shape | what |
|---|---|---|
| `Y_train` | (n_train, n_voxels) float | betas for the training images, one row per image, repeats averaged, z-scored per voxel |
| `img_train` | (n_train, 224, 224, 3) uint8 | training images, plain RGB 0-255, same order as `Y_train` |
| `Y_val` | (n_val, n_voxels) float | small held-out set of non-pairmate images, used for the health check |
| `img_val` | (n_val, 224, 224, 3) uint8 | |
| `Y_test` | (n_test, n_voxels) float | betas for the pairmate images |
| `img_test` | (n_test, 224, 224, 3) uint8 | |
| `pair_idx` | (n_pairs, 2) int | which rows of `Y_test` are pairmates |
| `test_names` | (n_test,) str | image names, used in the output csv |

Same voxels in the same order in all three `Y_` arrays. Do not ImageNet-normalise the images, the
code does that. Run `check_inputs.py` on it before training.

## If something goes wrong

| you see | what to do |
|---|---|
| `PROBLEM: ... has N lines but betas has M rows` | `trials.csv` needs one line per row of the betas, same order |
| `PROBLEM: betas look sideways` | your betas are voxels x trials, transpose and save again |
| `PROBLEM: pair image '...' is never in trials.csv` | a name in `pairs.csv` does not match `trials.csv`. Check spelling, the extension, and full path vs bare filename |
| `PROBLEM: can't find the image ...` | `--images` is pointing at the wrong folder |
| `PROBLEM: ... test (pairmate) images also appear in train/val` | a pairmate image got into training, so the result would be meaningless. Only happens if you built the npz yourself |
| `PROBLEM: pretrained base ... is missing` | run `./download_checkpoints.sh` |
| `ModuleNotFoundError: dinov2` or `models` | run the scripts from inside the repo folder, they add themselves to the path |
| torch.hub cannot download | run `./get_dinov2.sh` on a login node, and set `TORCH_HUB` in `config.sh` |
| `this checkpoint was trained against base ... which is not there now` | the base file moved, pass `--base` pointing at it |
| val retrieval near chance | the encoder did not fit. Check the betas are in the right order with the right mask, and that you have enough training images |
| `sbatch: error: invalid account` | fix `SLURM_ACCOUNT` in `config.sh`, or blank it if your cluster does not use accounts |

## What is in here

| file | what it is |
|---|---|
| `config.sh` | the only place you set paths and Slurm settings |
| `submit.sh` | queues the whole pipeline |
| `download_checkpoints.sh`, `get_dinov2.sh` | fetch the base weights and the backbone |
| `make_inputs.py` | builds the input file from your betas, csvs and images |
| `check_inputs.py` | checks an input file and explains what is wrong |
| `train_encoder.py` | fits the encoder to your subject |
| `eval_2afc.py` | 2AFC, CPD, retrieval and controls |
| `make_pred_cache.py` | caches one prediction per candidate image |
| `score_trials.py` | scores trials from that cache, no GPU needed |
| `ube/online.py` | the scorer you call from your own real time loop |
| `ube/load.py` | builds the model and loads the weight files |
| `ube/cache.py` | builds and reads the prediction cache |
| `realtime/rtcloud_example.py` | template for a live scan loop |
| `extras/` | base pretraining, and notes on what worked and what did not |
| `tools/export_base.py` | turns a big pickled encoder into the small weight files |
| `models/`, `dinov2/` | model code, needed to load the encoder, do not edit |

`make_inputs_bixby.py` reads our own lab's data directly and its paths will not exist for you. It
is kept as a worked example of going from raw per-session betas to an input file.

Questions: Akash (ab4736@princeton.edu)
