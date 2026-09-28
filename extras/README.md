# extras

This directory contains the code for pretraining a base encoder on NSD from scratch, which is not
required in order to run a pairmate analysis and is included for anyone who wants to reproduce or
modify the pretraining itself. Pretraining requires the full NSD dataset and a substantial amount
of GPU time, whereas the pretrained bases can simply be downloaded, so most users will not need
anything in here.

## What the contrastive objective is, and why it is the default

The base encoder can be trained in either of two ways, and the difference between them is the main
methodological point of this repository.

The original objective is reconstruction. For each image the encoder predicts a brain pattern, and
training minimises the difference between that prediction and the measured pattern, so the encoder
is rewarded for producing predictions that are accurate in an absolute sense. This works well, but
it does not explicitly ask the encoder to make the prediction for one image distinguishable from
the prediction for a different image, which is precisely what a pairmate comparison depends on.

The contrastive objective adds a second requirement on top of the reconstruction loss. Within each
training batch, the prediction for a given image must match that image's own measured pattern more
closely than it matches the measured pattern of any other image in the batch. Training therefore
penalises predictions that are accurate on average but insufficiently specific to the individual
image, which is the failure mode that matters most when two candidate images are nearly identical.
A small entropy term is included alongside it, which discourages the comparison from being
dominated by a single competing image within the batch.

In our testing this change improved discrimination for every NSD subject we evaluated, and for our
own subject as well, on both forced choice and retrieval. For that reason `ube_base_infonce.pt` is
the default base, and `ube_base_recon.pt` is provided as the matched reconstruction-only baseline
for anyone who wants to compare the two directly.

## Pretraining on NSD

Two scripts are provided, and they differ in the respect described below.

### nsd/, the original training code

The script `extras/nsd/train_encoder.py` is the Irani lab's own base training script, reproduced
essentially unchanged. The only modifications are to the import paths, to reading the data and
save directories from environment variables so that it runs on any machine, and the addition of a
flag that runs a few batches as a quick check. This is the code that produced the
reconstruction-only base, so it is the appropriate starting point for conventional pretraining.

```bash
cd extras/nsd
export UBE_NSD_DIR=/path/to/nsd_data/        # see the data section below
export UBE_SAVE_DIR=/path/to/save/
UBE_SMOKE=1 python train_encoder.py          # 3 batches, just checks it runs
python train_encoder.py                      # the real run, hours on one gpu
```

It writes a full pickled model into `UBE_SAVE_DIR`, which can be converted into the compact weight
format using `tools/export_base.py` if you want to distribute it or use it with
`train_encoder.py --base`. It also requires `tensorboardX`, which is listed in `requirements.txt`.

### train_base_contrastive.py, the contrastive version

This script trains the same model with the contrastive objective described above. It exists as a
separate script rather than as an additional flag on the original for a specific technical reason.
The original data loader draws a different random sample of 5,000 voxels for every item, and it
also allows a single batch to contain items from different subjects. Because the contrastive term
compares the items within a batch against one another, it is only meaningful when all of those
items are described in the same voxel space. Adding the term to the original loop would therefore
have compared patterns that were defined over different sets of voxels, which would produce a
number without producing a meaningful one.

This version accordingly constructs batches so that every item in a batch comes from the same
subject and uses the same sampled voxels. Everything else, including the model, the reconstruction
loss and the image augmentation, follows the original.

```bash
python extras/train_base_contrastive.py --data_dir /path/to/nsd_data --objective infonce --out mybase.pt
python extras/train_base_contrastive.py --data_dir /path/to/nsd_data --smoke --out /tmp/x.pt
```

Passing `--objective recon` performs a reconstruction-only run using the same batching scheme,
which is the appropriate matched control when comparing the two objectives against each other. The
script writes the compact weight format directly, so its output can be passed straight to
`train_encoder.py --base`.

It is worth being clear about what has and has not been verified here. The `--smoke` path has been
tested and works, in that it loads NSD, builds the model, takes three optimisation steps and saves
a checkpoint in the expected format. No base has been trained to completion using this script,
however, and the exact hyperparameters and batching used to produce the contrastive checkpoint that
is distributed here were not recorded at the time. This script therefore reproduces the objective
rather than that particular checkpoint, and anyone retraining should keep their own reconstruction
and contrastive runs matched to each other and draw comparisons only between those two.

## Other things worth knowing

Two further results from our own testing are worth recording, since both affect how the analysis
should be run rather than how the encoder is trained.

Initialising a new subject's voxel embeddings from the most similar NSD voxel, rather than at
random, improved the fit for one of our subjects. Doing so requires a voxel correspondence that has
to be constructed separately, and the warm starting section of the main README describes what is
involved.

For real-time use, scoring approximately six to nine seconds after stimulus onset worked best, and
widening the analysis window did not allow earlier decoding, because a wider window necessarily
includes volumes acquired before the response has developed.

Finally, decoding the measured response into CLIP space in order to distinguish pairmates is the
more obvious approach, and it performed clearly worse than running the encoder forward, on the same
trials. Pairmate images occupy nearly the same position in CLIP space, so the information needed to
separate them is substantially reduced before the comparison is made. Lower-level image features
performed better than CLIP in this respect but still did not match the encoder.
