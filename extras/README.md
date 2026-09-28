# extras

Boilerplate from other things we tried. None of this is needed for a normal 2AFC test. It is
here so you can build on it, and so the things that did not work are written down instead of
being tried again.

## train_base.py

Pretrains the base encoder on NSD from scratch. Needs the full NSD dataset and days of GPU
time. See section 3 of HUGGINGFACE_UPLOAD.md for the data and the file layout it expects.

```
python extras/train_base.py --data_dir /path/to/nsd_data --objective infonce --out mybase.pt
python extras/train_base.py --data_dir /path/to/nsd_data --smoke --out /tmp/x.pt   # 3 steps, checks it runs
```

`--objective recon` gives the plain reconstruction baseline, `--objective infonce` adds the
contrastive term. Those two switches are the difference between the two base checkpoints we
ship. It writes the same small weight format as the downloaded bases, so a base you train here
drops straight into `train_encoder.py --base`.

The `--smoke` run above has been tested and works: it loads NSD, builds the model, takes 3 steps
and saves a 9.6 MB checkpoint. A full run has **not** been done from this script, so treat the
hyperparameters as a starting point rather than a recipe.

One design note. At base training time each step samples 5,000 random voxels, and different
subjects have different voxels, so an in-batch contrastive term is only meaningful if every
item in the batch shares the same voxels and the same subject. This script builds batches that
way. The exact batching used for the shipped checkpoint was not recorded, so if you retrain,
keep your own recon and infonce runs matched and compare those to each other.

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
