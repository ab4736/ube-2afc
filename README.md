# ube-2afc: pairmate discrimination from fMRI

This repository provides code for testing whether a person's measured fMRI response can be used to
determine which of two near-identical images they were viewing. It supports both offline analysis
of a completed dataset and online scoring during an ongoing scan.

The encoder it is built on is the Universal Brain Encoder (UBE) from the Irani lab, described in
[Beliy et al.](https://arxiv.org/abs/2406.12179), whose code is available at
[WeizmannVision/brainit-fmri](https://github.com/WeizmannVision/brainit-fmri). What this repository
adds is the pairmate side of the analysis: the forced-choice and CPD scoring, the procedure for
fitting the encoder to a new subject, and the machinery for running the scoring live during
acquisition. Image reconstruction is not included here, and their repository should be used for
that purpose.

## How it works

Two pairmate images are almost the same picture, so the usual approach of decoding the brain data
into CLIP features and asking which image it matches does not work well. Both images have decoded
CLIP embeddings that are hard to discriminate, due to information loss from the decoder being used.

This implementation therefore works in the opposite direction. Rather than decoding the measured
response into image features, it uses the encoder to predict, for each of the two candidate images,
the brain pattern that image should have produced. The trial is then scored by asking which of the
two predicted patterns the measured pattern more closely resembles.

```
pred_A = encoder(image A)
pred_B = encoder(image B)
correct if corr(measured beta, pred_A) > corr(measured beta, pred_B)
```

Chance performance is 0.50. Because the image is never reconstructed, the comparison does not
depend on recovering image content that the two pairmates share.

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

3. Open `config.sh` and fill in the settings at the top, which are the path to the Python
interpreter you just created and, if your cluster requires them, your Slurm account and partition.
No other file in the repository contains a machine-specific path, so this is the only file that
needs to be edited for a new cluster.

4. Download the pretrained weights:

```bash
./download_checkpoints.sh
```

This retrieves the encoder weights, which are roughly 10 MB each, along with the DINOv2 backbone,
which is 1.2 GB. Once these are in place the code does not require network access again, which
matters because compute nodes on most clusters cannot reach the internet. If the HuggingFace
repository has been made private, you will need to run `huggingface-cli login` before downloading.

Running the analysis requires a Slurm cluster with one GPU of 16 GB or more.

## General information

This repository contains scripts for

- Building an input file from your data (`make_inputs.py`)
- Checking that input file before spending time on a training run (`check_inputs.py`)
- Fitting the encoder to your subject (`train_encoder.py`)
- Scoring pairmate 2AFC and CPD (`eval_2afc.py`)
- Caching predictions so that trials can be scored immediately (`make_pred_cache.py`)
- Scoring trials individually, either after a session or during one (`score_trials.py`,
  `ube/online.py`)

For most purposes it is sufficient to run `./submit.sh NAME`, which executes the middle four steps
in order as a single Slurm job.

The `extras/` directory contains the code for pretraining a base encoder from scratch, together
with notes recording which methodological variations improved performance and which did not.

## Pretrained models

Pretraining can be skipped entirely, since `download_checkpoints.sh` retrieves the following
pretrained bases:

- `ube_base_infonce.pt` was pretrained with the contrastive objective, and is the default and the
  one that should normally be used.
- `ube_base_recon.pt` uses the same architecture and training data but a reconstruction loss alone,
  and is provided as a matched baseline for anyone wishing to compare the two objectives.
- `ube_base_original.pt` is the earlier base from which this work started.

The base can be changed using the `BASE=` line in `config.sh`. These files are small because the
DINOv2 backbone is not stored inside them; the backbone is unmodified and is loaded separately.

## Preparing your data

Three files are required, along with the images themselves.

The first is a matrix of betas, with one row per trial and one column per voxel. Any masking should
be applied beforehand so that only the voxels of interest are included. Single-trial betas are
acceptable, as repeated presentations of the same image are averaged automatically. The matrix can
be saved from Python using `np.save("betas.npy", betas)`, or from MATLAB using
`save('betas.mat','betas','-v7')`, provided that the file contains only that one variable.

The second is `trials.csv`, which records which image was presented on each trial. It must contain
one line per row of the beta matrix, in the same order, and must include a column named `image`. A
`session` column is optional but recommended, because when it is present each voxel is z-scored
within session, which removes session-level scaling differences.

```
image,session
scene_1023.png,1
pair_3_1.jpg,1
scene_88.jpg,1
scene_1023.png,2
pair_3_2.jpg,2
```

The third is `pairs.csv`, which lists the pairmate pairs, one per line. The image names must match
those in `trials.csv` exactly.

```
image_a,image_b
pair_3_1.jpg,pair_3_2.jpg
pair_4_1.jpg,pair_4_2.jpg
```

The images listed in `pairs.csv` become the test set, and they are removed from the training set
automatically, so the encoder is never exposed to them during fitting.

The images themselves can be ordinary .jpg or .png files of any size. They are resized to 224x224
during preparation, and non-square images are stretched rather than cropped.

Every image that is not a pairmate is used for training, so a larger stimulus set generally
produces a better fit. Roughly 2,000 images works well. Fewer than about 1,000 will still run, but
the encoder will fit the subject less accurately.

The input file is then built as follows, where `--name` is an identifier for the dataset, typically
a subject ID. The `--images` argument can be omitted if `trials.csv` already contains full paths.

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

The resulting file should then be checked. This takes only a few seconds and identifies the kinds
of formatting problems that would otherwise only become apparent after waiting in the queue for a
training job.

```bash
$PY check_inputs.py data/sub-08.npz
```

The check reports how many images and voxels were found. A final line reading `ALL GOOD` indicates
that the file is usable. Any line beginning with `PROBLEM:` describes something that must be
corrected before proceeding, whereas lines beginning with `warning:` describe conditions that are
worth knowing about but do not prevent the analysis from running.

## Running the whole thing

```bash
./submit.sh sub-08
```

This queues a single job that checks the input file, fits the encoder to the subject, which takes
approximately twenty minutes, scores 2AFC and CPD, and builds the prediction cache used for online
scoring. Job status can be checked with `squeue -u $USER`, and progress can be followed with
`tail -f logs/ube_2afc_*.log`.

The sections below describe the individual steps that this job performs, for cases where they need
to be run separately.

## Fitting the encoder to your subject

During fitting the pretrained base remains frozen, and the only parameters that are trained are the
per-voxel embeddings for the subject, one vector per voxel. Because the number of trained
parameters is small, fitting is fast and does not require a large stimulus set.

```bash
$PY train_encoder.py --data data/sub-08.npz --tag sub-08_infonce \
    --base checkpoints/ube_base_infonce.pt --lam 1 --went 0.1
```

- Setting `--lam 1 --went 0.1` enables the contrastive objective, which is the configuration that
  improves pairmate discrimination.
- Setting `--lam 0 --went 0` trains a plain reconstruction encoder, which is useful if you want to
  compare the two objectives directly.
- The resulting file is a few megabytes rather than 1.2 GB, because it stores only the voxel
  embeddings together with a reference to the base they were trained against.
- Fitting is reproducible, in that the same `--seed` applied to the same input file will produce
  identical results.

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

The evaluation prints a single summary block in the following form, with your own values in place
of the Xs:

```
  pairmate 2AFC   0.XXX  (nn/NN)  95% CI [0.XX, 0.XX]  chance 0.5
  CPD             mean +0.XXX  median +0.XXX  CPD>0 0.XXX
  retrieval       top-1 0.XXX  (1 of NN, chance 0.0XX)
  val retrieval   top-1 0.XXX  (1 of NNN, chance 0.00X)
  controls        shuffled_voxels 0.5X  mean_beta 0.50  wrong_image 0.1X
```

- Pairmate 2AFC is the primary measure. It reports the proportion of trials on which the measured
  pattern was more similar to the prediction for the image that was shown than to the prediction
  for its pairmate. Each pair contributes two trials, one with each image treated as the presented
  image. The confidence interval is obtained by resampling pairs.
- CPD is the signed projection of the measured pattern onto the line joining the two predictions. A
  value of +1 indicates that the measured pattern lies exactly on the prediction for the presented
  image, 0 indicates that it lies midway between the two, and -1 indicates that it lies on the
  prediction for the pairmate.
- Retrieval is a more demanding version of the same comparison, in which each measured pattern is
  compared against the predictions for all test images rather than against its pairmate alone.
- Validation retrieval applies the same procedure to ordinary non-pairmate images, and serves as a
  check on the fit. If it is close to chance, the encoder has not fit the subject, and the 2AFC
  value should not be interpreted.
- The control measures should all fall near 0.50, although `wrong_image` may fall below it. The
  `shuffled_voxels` control permutes the voxels of the predictions, `mean_beta` replaces the
  measured pattern with the average pattern across trials, and `wrong_image` substitutes the
  prediction for an unrelated scene. Elevated values on any of these indicate that information is
  leaking into the comparison and that the result is not trustworthy.

The evaluation also writes `results/sub-08_infonce_trials.csv`, which contains one row per trial
recording both correlations, whether the trial was scored correct, and the CPD value, in a form
that can be opened directly in a spreadsheet program.

## Scoring trials as they happen (neurofeedback)

Online use has two components, the first of which is also useful on its own for obtaining per-trial
values after a session has finished.

The first component is the prediction cache, which is built before the scan and is produced
automatically by `submit.sh`. It runs the encoder once for each candidate image and stores the
resulting predicted patterns, so that no GPU computation is required while the scan is running.

```bash
$PY make_pred_cache.py --data data/sub-08.npz --enc checkpoints/sub-08_infonce.pth \
    --out cache/sub-08_preds.npz
```

The second component is the per-trial scoring itself. For a session that has already finished,
`score_trials.py` scores each trial in turn, where `shown_foil.csv` contains the columns `shown`
and `foil`, one line per trial, in the same order as the rows of the beta matrix. Passing
`--causal_z` normalises each trial using only the trials observed up to that point, which is the
appropriate choice when reproducing the conditions of an online analysis.

```bash
$PY score_trials.py --cache cache/sub-08_preds.npz --betas trial_betas.npy \
    --trials shown_foil.csv --out results/pertrial.csv
```

For live use, the scorer can be called directly from within an existing acquisition loop. It is
implemented in pure numpy and requires no GPU, so the time it takes is negligible relative to
acquisition.

```python
from ube.online import PairmateScorer
scorer = PairmateScorer.from_cache("cache/sub-08_preds.npz")

scorer.update_running_stats(beta)     # feed each trial's beta in as it arrives
b = scorer.causal_z(beta)             # normalise using only the past
out = scorer.score(b, shown="pair_3_1.jpg", foil="pair_3_2.jpg", z=False)
print(out["cpd"], out["correct"])     # send out["cpd"] back as feedback
```

A worked version of this loop is provided in `realtime/rtcloud_example.py`. The one component that
must be supplied locally is the step that converts the volumes acquired so far into a single beta
for the current trial, since this differs between acquisition setups. Fitting a causal GLM for this
purpose performs considerably better than simply averaging the volumes within the response window,
which in our testing performed close to chance.

Two further observations about timing are worth noting. Scoring is best performed approximately six
to nine seconds after stimulus onset, since accuracy peaked at around nine seconds and declined
when scoring was delayed further. Widening the window does not allow earlier decoding, because a
wider window necessarily incorporates volumes acquired before the response has developed, which
delays rather than advances the point at which the measure becomes informative.

One caveat applies specifically to CPD in the online setting. The 2AFC measure is insensitive to
the overall scaling of the betas, whereas CPD is not, so changing the normalisation scheme shifts
the CPD values. CPD values should therefore only be compared across analyses that used the same
normalisation.

## Warm starting (optional)

By default the voxel embeddings for a new subject are initialised randomly. They can instead be
initialised from the embedding of the most similar NSD voxel, which improved the fit for one of our
subjects.

```bash
WITH_WARMSTART=1 ./download_checkpoints.sh      # adds a 324 MB file
$PY train_encoder.py --data data/sub-08.npz --tag sub-08_warm \
    --base checkpoints/ube_base_infonce.pt \
    --warm_embed checkpoints/nsd_voxel_embed_infonce.pt --warm_map my_voxel_to_nsd.npy
```

Using this option requires constructing `my_voxel_to_nsd.npy`, an integer array with one entry per
voxel of the subject giving the index of the corresponding NSD voxel. Producing it requires
bringing the subject's voxels and the NSD voxels into a common space, such as fsaverage, and taking
nearest neighbours. If such a correspondence is not available, this option can be skipped, as
random initialisation performs adequately.

## Pretraining the base yourself

Pretraining is not necessary for running the analysis, and it requires the full NSD dataset as well
as a substantial amount of GPU time. For those who do want to carry it out, the preprocessed NSD
data is available from the same HuggingFace repository:

```bash
WITH_NSD=1 ./download_checkpoints.sh      # 17 GB, and it is NSD data, so their terms apply
```

The directory `extras/nsd/` contains the original Irani lab training script, reproduced with only
minimal changes, and is the appropriate starting point for conventional pretraining.

The script `extras/train_base_contrastive.py` is a modified version that adds the contrastive term.
It is provided as a separate script rather than as an option on the original because the
contrastive term requires batches in which every item shares the same sampled voxels and the same
subject, a condition that the original data loader does not satisfy. Both are described in
`extras/README.md`.

## The input file format

If you prefer to construct `data/NAME.npz` directly rather than using `make_inputs.py`, it is a
numpy `.npz` archive containing the following arrays:

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

The same voxels must appear in the same order in all three `Y_` arrays. The images should not be
ImageNet-normalised beforehand, as this is handled internally. It is worth running
`check_inputs.py` on any file constructed this way before beginning training.

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
| torch.hub cannot download | you are missing the backbone. run `./download_checkpoints.sh`, which fetches it, or `./get_dinov2.sh` on a login node |
| `this checkpoint was trained against base ... which is not there now` | the base file moved, pass `--base` pointing at it |
| val retrieval near chance | the encoder did not fit. Check the betas are in the right order with the right mask, and that you have enough training images |
| `sbatch: error: invalid account` | fix `SLURM_ACCOUNT` in `config.sh`, or blank it if your cluster does not use accounts |

## What is in here

| file | what it is |
|---|---|
| `config.sh` | the only place you set paths and Slurm settings |
| `submit.sh` | queues the whole pipeline |
| `download_checkpoints.sh` | fetches the weights and the backbone from hugging face |
| `get_dinov2.sh` | fallback way to get the backbone through torch.hub, if you did not download it |
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

Questions: Akash (ab4736@princeton.edu)
