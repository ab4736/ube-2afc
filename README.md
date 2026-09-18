# UBE pairmate 2AFC

Code for doing pairmate discrimination (2AFC and CPD) with the Irani lab Universal Brain Encoder (UBE).

You give it someone's fMRI betas, the images they saw, and a list of which images are pairmates. It gives you back, for every pairmate trial, whether the brain pattern "picked" the right image, plus a CPD score.

Short version of the idea: instead of turning the brain data into CLIP features and asking which image it's closer to (the usual MindEye direction), go the other way. The encoder looks at an image and predicts what the brain pattern for it should look like. So for a trial where someone saw image A, we predict a pattern for A and a pattern for its pairmate B, and check which prediction the real brain pattern is more similar to. If it's A, that trial is correct.

For pairmates that are designed to be really similar, this works a lot better than the decoder route, because CLIP mostly throws away the small differences between two versions of the same scene.

Paper for the encoder: [The Wisdom of a Crowd of Brains: A Universal Brain Encoder](https://arxiv.org/abs/2406.12179) (Beliy, Wasserman, Zalcher, Irani).

---

## The steps, start to finish

1. Get onto Della
2. Get the code into your own space
3. Make your input file
4. Check the input file
5. Run it
6. Look at the results

If you just want to see it work first, use the Bixby subjects (sub-06 / sub-07). Skip step 3 for those, it's automatic.

---

## 1. Get onto Della

You need a Della account on the `knorman` group. Wanjia, Dhairyya and Aidan already have one. Shruti, you'll need to ask Ken to get added.

Two ways in:

- Easiest: go to https://mydella.princeton.edu in a browser (needs the Princeton VPN if you're off campus). Log in, then click **Clusters > Della Shell Access** at the top. That gives you a terminal in the browser. You can also browse, upload and download files there with **Files > Home Directory**.
- Or from your own terminal: `ssh YOUR_NETID@della.princeton.edu`

Everything below gets typed into that terminal. Anything in CAPS like `YOUR_NETID` is something you replace.

## 2. Get the code into your own space

Change `YOUR_FOLDER` below to your folder under KNORMAN (for example `agoeschel`, `ds2157`). Two ways, pick one.

**Option A, from GitHub** (if you've been added to the repo):

```bash
mkdir -p /scratch/gpfs/KNORMAN/YOUR_FOLDER
cd /scratch/gpfs/KNORMAN/YOUR_FOLDER
git clone https://github.com/GITHUB_OWNER/ube-2afc.git
cd ube-2afc
```

If it asks for a GitHub password and you don't know what to put, just use option B instead.

**Option B, copy it straight from Akash's folder on Della** (no GitHub needed):

```bash
mkdir -p /scratch/gpfs/KNORMAN/YOUR_FOLDER
rsync -a --exclude 'data/*' --exclude 'checkpoints/*' --exclude 'results/*' --exclude 'logs/*' \
    /scratch/gpfs/KNORMAN/ab4736/ube-2afc/ /scratch/gpfs/KNORMAN/YOUR_FOLDER/ube-2afc/
cd /scratch/gpfs/KNORMAN/YOUR_FOLDER/ube-2afc
```

From now on, always be inside that folder when you run things (the last `cd` line above puts you there). If you log out and come back, run `cd /scratch/gpfs/KNORMAN/YOUR_FOLDER/ube-2afc` again.

Then set this once per login. It's just a shortcut to the Python everyone uses, so you don't have to install anything:

```bash
PY=/scratch/gpfs/KNORMAN/qanguyen/Brain-IT/brain-it/bin/python
```

## 3. Make your input file

Everything runs off one file, `data/NAME.npz`, where NAME is whatever you want to call the dataset (a subject ID works well).

### For Bixby sub-06 / sub-07

Nothing to do, `run_pipeline.slurm` builds it for you. (If you want to build it by hand: `$PY make_inputs_bixby.py --sub sub-06`)

### For your own data

You need three files plus your images. Put them wherever you want, you'll give the paths.

**File 1: the betas.** A table of numbers: one row per trial, one column per voxel.

- Only include the voxels you want to use (apply your mask first).
- Single-trial betas are fine. If the same image was shown more than once, keep every trial as its own row, the script averages them for you.
- Save it from Python with `np.save("betas.npy", betas)`, or from MATLAB with `save('betas.mat', 'betas', '-v7')` (only that one variable in the file).

**File 2: `trials.csv`.** Says which image was on screen for each row of the betas, in the same order. One line per row. It needs a column called `image`. A `session` column is optional but recommended: if you have it, each voxel gets z-scored within each session.

```
image,session
notspecial_1023.png,1
pair_3_1.jpg,1
special_88.jpg,1
notspecial_1023.png,2
pair_3_2.jpg,2
...
```

The image column can be just the file name (then tell the script which folder the images are in), or the full path to the file.

**File 3: `pairs.csv`.** One line per pairmate pair, columns `image_a` and `image_b`. The names have to be written exactly the same way as in `trials.csv`.

```
image_a,image_b
pair_3_1.jpg,pair_3_2.jpg
pair_4_1.jpg,pair_4_2.jpg
...
```

These pairmate images become the test set. They're automatically taken out of training, so the encoder never sees them.

**The images:** normal .jpg / .png files, any size (they get resized to 224x224; if they aren't square they get stretched a bit, that's fine).

Then run this (all one command):

```bash
$PY make_inputs.py --name NAME --betas /path/to/betas.npy --trials /path/to/trials.csv \
    --pairs /path/to/pairs.csv --images /path/to/image/folder
```

Leave out `--images` if `trials.csv` already has full paths. This takes a few minutes and writes `data/NAME.npz`. If something's off with your files it stops and tells you what (see "If something goes wrong" below).

How much data you need: every image that isn't a pairmate is used to train, so more is better. Bixby has about 2100 training images from its three pretraining sessions. Below ~1000 it will still run but probably fits worse.

## 4. Check the input file

```bash
$PY check_inputs.py data/NAME.npz
```

It prints how many images and voxels it found. If the last line is `ALL GOOD`, you're fine. If it says `PROBLEM:` it tells you what's wrong. `warning:` lines are things to know about, but it'll still run.

The pipeline runs this check by itself too, and stops before training if there's a problem. So this step is just so you find out in 10 seconds instead of after waiting in the queue.

## 5. Run it

```bash
sbatch run_pipeline.slurm NAME
```

(For Bixby: `sbatch run_pipeline.slurm sub-06`)

That sends it to a GPU on Della. It prints a job number, for example `Submitted batch job 14111806`. It does three things in a row: checks the inputs, trains the encoder for your subject (about 20 min), then runs the 2AFC / CPD evaluation (about a minute).

To see if it's still running:

```bash
squeue -u YOUR_NETID
```

If your job is in the list it's still going (`PD` = waiting for a GPU, `R` = running). If it's not in the list, it's finished. Waiting for a GPU can take anywhere from a few seconds to a while depending on how busy Della is.

To watch the progress while it runs (`Ctrl+C` to stop watching, the job keeps going):

```bash
tail -f logs/ube_2afc_JOBNUMBER.log
```

To cancel it: `scancel JOBNUMBER`

## 6. Look at the results

At the bottom of `logs/ube_2afc_JOBNUMBER.log` you'll see something like this (this is sub-06):

```
  pairmate 2AFC   0.923  (24/26)  95% CI [0.81, 1.00]  chance 0.5
  CPD             mean +0.507  median +0.498  CPD>0 0.923
  retrieval       top-1 0.808  (1 of 26, chance 0.038)
  val retrieval   top-1 0.386  (1 of 233, chance 0.004)
  controls        shuffled_voxels 0.58  mean_beta 0.50  wrong_image 0.15
```

What each line means:

- **pairmate 2AFC**: the main number. Fraction of trials where the real brain pattern was more similar to the right image's prediction than to the pairmate's. 0.5 is chance, 1.0 is perfect. Each pair gives two trials (once with A as the shown image, once with B). The 95% CI is a range for how much this could move around with different pairs.
- **CPD**: same geometry as the grant (Fig. 2), just using predicted brain patterns instead of CLIP. Take the line between the two predictions. +1 means the real brain pattern sits right on the shown image's prediction, 0 means exactly halfway, -1 means it sits on the pairmate's. `CPD>0` is the fraction of trials on the right side of halfway.
- **retrieval**: a harder version. Each real pattern against the predictions for all the test images, not just its pairmate. Chance is 1 over the number of test images.
- **val retrieval**: the same thing but on ordinary held-out (non-pairmate) images. This one is a health check: if it's close to chance, the encoder didn't learn your subject and the 2AFC number shouldn't be trusted.
- **controls**: things that should come out around 0.5 if nothing is leaking. `shuffled_voxels` scrambles the predictions, `mean_beta` uses the average brain pattern instead of the real one, `wrong_image` uses some other random image's prediction (this one can be below 0.5). If any of these are high, something is wrong with the data, tell me.

You also get two files in `results/`:

- `results/NAME_infonce_trials.csv`: one row per trial. Columns: the shown image, the pairmate, the similarity to each prediction (`r_shown`, `r_foil`), whether it was correct (1 or 0), and the CPD. Opens in Excel / Google Sheets (download it through MyDella under Files).
- `results/NAME_infonce.json`: the summary numbers.

The trained encoder is saved as `checkpoints/NAME_infonce.pth` (~1.2 GB). You only need it again if you want to re-run the evaluation without retraining:

```bash
$PY eval_2afc.py --data data/NAME.npz --enc checkpoints/NAME_infonce.pth
```

(That part needs a GPU too. For a quick one-off it's fine to put it in a copy of `run_pipeline.slurm` with the training line deleted.)

---

## Where everything is

Things you use but don't need to touch (everyone can read these):

| what | where |
|---|---|
| This code (Akash's copy, don't run things in here) | `/scratch/gpfs/KNORMAN/ab4736/ube-2afc/` |
| Python to use | `/scratch/gpfs/KNORMAN/qanguyen/Brain-IT/brain-it/bin/python` |
| UBE base encoder (the scripts point here already) | `/scratch/gpfs/KNORMAN/ab4736/brainit-fmri/results/saved_models/encoder_ch128_base_nce20_ent.pth` |
| Bixby betas, sub-06 | `/scratch/gpfs/KNORMAN/ab4736/brainit-fmri/sub-06_roi_vox_all_sessions.pkl` |
| Bixby betas, sub-07 | `/scratch/gpfs/KNORMAN/ab4736/brainit-fmri/sub-07_roi_vox_all_sessions.pkl` |
| Bixby images | `/scratch/gpfs/KNORMAN/miguelp/mindeye_offline/all_stimuli/` |
| Already-trained Bixby encoders (to skip training) | `/scratch/gpfs/KNORMAN/ab4736/ube-2afc/checkpoints/sub-0{6,7}_infonce.pth` |

Inside your copy of the folder:

| file / folder | what it is |
|---|---|
| `README.md` | this |
| `run_pipeline.slurm` | the one you run. Check inputs, train, evaluate |
| `make_inputs.py` | builds `data/NAME.npz` from your betas + trials.csv + pairs.csv + images |
| `make_inputs_bixby.py` | same thing but reads the Bixby pickle files directly |
| `check_inputs.py` | looks at an input file and tells you if something's wrong |
| `train_encoder.py` | fits the encoder to your subject |
| `eval_2afc.py` | the 2AFC / CPD evaluation |
| `models/`, `dinov2/` | the encoder's code. Needed to load it, don't edit |
| `data/` | your input files end up here |
| `checkpoints/` | trained encoders end up here |
| `results/` | the csv and json results end up here |
| `logs/` | the job logs (what gets printed while it runs) end up here |

## The input file format (if you want to build it yourself)

If you'd rather make `data/NAME.npz` your own way instead of using `make_inputs.py`, it's a numpy `.npz` with these arrays:

| key | shape | what |
|---|---|---|
| `Y_train` | (n_train, n_voxels) float | betas for the training images, one row per image (repeats averaged), z-scored per voxel |
| `img_train` | (n_train, 224, 224, 3) uint8 | the training images, plain RGB 0-255, same order as `Y_train` |
| `Y_val` | (n_val, n_voxels) float | a small held-out set of non-pairmate images, only used for the health check (we use 10%) |
| `img_val` | (n_val, 224, 224, 3) uint8 | |
| `Y_test` | (n_test, n_voxels) float | betas for the pairmate images |
| `img_test` | (n_test, 224, 224, 3) uint8 | |
| `pair_idx` | (n_pairs, 2) int | which rows of `Y_test` are pairmates, one line per pair |
| `test_names` | (n_test,) str | names of the test images, only used in the output csv (optional) |

Same voxels, in the same order, in all three `Y_` arrays. Don't do ImageNet normalization on the images, the scripts do it. Run `check_inputs.py` on it before training.

## If something goes wrong

| you see | what it means / what to do |
|---|---|
| `PROBLEM: ... has N lines but betas has M rows` | `trials.csv` needs exactly one line per row of the betas, in the same order |
| `PROBLEM: betas look sideways` | your betas are voxels x trials. Flip it (`betas.T` in Python, `betas'` in MATLAB) and save again |
| `PROBLEM: pair image '...' is never in trials.csv` | a name in `pairs.csv` doesn't exactly match any name in `trials.csv`. Check spelling, the extension (.jpg vs .png), and whether one has a full path and the other doesn't |
| `PROBLEM: can't find the image ...` | the image file isn't there. Check `--images` points at the right folder |
| `PROBLEM: ... test (pairmate) images also appear in train/val` | a pairmate image got into the training set, so the result would be meaningless. Only happens if you built the npz yourself |
| `... has N variables, save just the betas matrix in it` | your .mat file has other stuff in it. Save only the betas: `save('betas.mat', 'betas', '-v7')` |
| `sbatch: error: ... invalid account` | you're not on the knorman Della group yet, ask Ken |
| `PROBLEM: data/NAME.npz doesn't exist` | make the input file first (step 3), and make sure the NAME matches |
| nothing in `logs/` and `squeue` shows nothing | the job didn't start. Make sure you ran `sbatch` from inside the folder |
| `CUDA out of memory` | ask Akash |
| val retrieval near chance | the encoder didn't fit your subject. Check the betas are right (right order, right mask), and that you have enough training images |

## How it works (the more technical bit)

The UBE base is DINOv2 (frozen) plus the UBE layers, trained on NSD. The version here was also trained with an InfoNCE term on top of the normal reconstruction loss, which makes it better at telling similar images apart.

To use it on a new person, the base stays frozen and we only train `voxel_embed`: one learned vector per voxel of that person. Each voxel's vector learns "what kind of image stuff this voxel responds to", and the frozen part does the rest. That's why it works with a couple thousand images instead of needing a whole NSD-sized dataset.

Training loss (in `train_encoder.py`): mean squared error between predicted and real betas, minus 0.1 x cosine similarity, plus an InfoNCE term (`--lam 1`) that pushes each prediction to match its own real beta better than the other betas in the batch, plus a small entropy term (`--went 0.1`). 40 epochs, batch 32, Adam, lr 1e-3, ±3 pixel random shifts on the images. For a plain reconstruction encoder use `--lam 0 --went 0`.

The similarity in the 2AFC is Pearson correlation across voxels, which is the same as the dot product after z-scoring each pattern. Don't swap it for plain cosine on the raw patterns, on sub-005 that dropped 2AFC from 0.935 to 0.855.

## Numbers we got

Bixby, pretraining sessions 01/02/03, 13 held-out scene pairs (26 trials), `union_mask` voxels:

| | 2AFC | 95% CI | mean CPD | retrieval |
|---|---|---|---|---|
| sub-06 | 0.923 | [0.81, 1.00] | +0.507 | 0.808 |
| sub-07 | 0.692 | [0.50, 0.85] | +0.246 | 0.500 |

Retraining from scratch with this code (default seed) gives exactly these numbers. With only 26 trials the CIs are wide, so small differences don't mean much.

For comparison, on sub-005 (the offline MST data, 31 pairs) this approach got 0.935, and the Brain-IT CLIP decoder got 0.71 on the same trials.

The ses-04/05 snap scenes for Bixby aren't included, their stimulus folder wasn't readable last I checked. Once the images are readable it's just a different input file.

## Working on the code

If you change something, do it on your own branch so you don't step on each other:

```bash
git checkout -b YOUR_NAME-whatever-youre-changing
# ...edit things...
git add -A
git commit -m "what you changed"
git push -u origin YOUR_NAME-whatever-youre-changing
```

Then open a pull request on GitHub. Data, checkpoints, results and logs are never uploaded (they're too big, and `.gitignore` skips them), only the code.

Questions: Akash (ab4736@princeton.edu)
