# extras

Boilerplate from other things we tried. None of this is needed for a normal 2AFC test. It is
here so you can build on it, and so the things that did not work are written down instead of
being tried again.

## pretraining the base on NSD

Two options in here, and they are different on purpose.

### nsd/ : the original training code

`extras/nsd/train_encoder.py` is the Irani lab's own base training script, copied over
essentially unchanged. The only edits are the import paths, reading the data and save
directories from environment variables so it runs anywhere, and a smoke flag. This is the code
that produced the reconstruction base we ship, so if you want to pretrain, start here.

```bash
cd extras/nsd
export UBE_NSD_DIR=/path/to/nsd_data/        # see the data section below
export UBE_SAVE_DIR=/path/to/save/
UBE_SMOKE=1 python train_encoder.py          # 3 batches, just checks it runs
python train_encoder.py                      # the real thing, days on one gpu
```

It writes a full pickled model into `UBE_SAVE_DIR`. Convert it to the small format with
`tools/export_base.py` if you want to share it or use it with `train_encoder.py --base`.

Needs `tensorboardX` (in requirements.txt).

### train_base_contrastive.py : our modified version

This is the contrastive variant, and it is a separate script rather than a flag on theirs for a
real reason. Their `EncDataset` samples a **different random set of 5,000 voxels for every
item**, and batches mix subjects. A contrastive term compares the items in a batch against each
other, so it is only meaningful if they are all in the same voxel space. Bolting one onto their
loop would silently compare patterns across mismatched voxels.

So this version builds batches differently: one subject per batch, one shared voxel sample per
batch. Everything else (the model, the reconstruction loss, the augmentation) matches.

```bash
python extras/train_base_contrastive.py --data_dir /path/to/nsd_data --objective infonce --out mybase.pt
python extras/train_base_contrastive.py --data_dir /path/to/nsd_data --smoke --out /tmp/x.pt
```

`--objective recon` gives a reconstruction run with the same batching, which is the matched
control if you want to compare objectives. It writes the small weight format directly, so the
output drops straight into `train_encoder.py --base`.

Be aware of what is and is not verified here. The `--smoke` path is tested and works: it loads
NSD, builds the model, takes 3 steps and saves a 9.6 MB checkpoint. **No base has been trained
to completion with this script**, and the exact hyperparameters and batching used for the
contrastive checkpoint we ship were never written down, so this reproduces the objective, not
that specific checkpoint. If you retrain, keep your own recon and contrastive runs matched and
compare those two to each other.

## what worked

- **The contrastive (InfoNCE + entropy) pretraining objective.** Won on 8 of 8 NSD subjects
  (2AFC 0.870 to 0.954, retrieval 0.356 to 0.695). On sub-06, fine-tuning from it took 2AFC
  from 0.808 to 0.923. This is the main win and it is the default.
- **Warm starting a new subject's voxel embeddings** from the nearest NSD voxel instead of
  random init. Helped on sub-005. Needs a voxel correspondence you have to build yourself, see
  the warm start section of the main README.
- **Scoring at 6 to 9 s after onset** for the real time case, and not widening the window.

## what did not work

Written down so nobody burns a week on them again.

- **Hard negative batch mining with a DINOv2 teacher** (the "Breaking the Batch Barrier" idea).
  Batches were verifiably about 10x harder in teacher space and it still did nothing:
  0.0079 *below* random batching over 8 seeds. Mining in **voxel space** instead was slightly
  positive (+0.0087, 6 of 8 seeds) but it does not survive a correction for the three variants
  tested, so treat it as unproven, not as a result.
- **Continuous per-voxel reliability weighting** instead of a binary reliability mask. Exactly
  zero effect over 3 seeds, despite the reliability values being genuinely spread out. The
  weighting reallocates the loss without changing what the model learns to tell apart.
- **Reinforcement learning / evolution strategies** on top of the contrastive objective. Reward
  saturated, no gain over InfoNCE.
- **Brain-JEPA style masking** (gradient-position init, cluster and predicted masking). Neutral.
- **Warm starting from a z-prior.** Failed, the session shift is too large.
- **Decoding into CLIP space to tell pairmates apart.** This is the obvious thing to try and it
  is much worse than the encoder direction: 0.71 for a real CLIP decoder vs 0.935 for the
  encoder on the same trials. Pairmates collapse to nearly the same CLIP vector, so the
  information is not there to decode. Low level (VGG) features do better than CLIP (0.81) but
  still lose to the encoder.
