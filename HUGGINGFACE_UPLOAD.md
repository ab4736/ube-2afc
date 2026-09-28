# what to put on hugging face

Everything here is for you (Akash), not for the people using the repo. It lists exactly which
files to upload, which files must NOT be uploaded, and what someone needs if they want to
pretrain the base encoder themselves.

Make a **model** repo (not a dataset repo) at `huggingface.co/new`, name it `ube-2afc`, and set it
to Private for now. Then upload the files below. The download script in the repo
(`download_checkpoints.sh`) points at `ab4736/ube-2afc` and expects these exact filenames.

## 1. upload these (the model weights)

These are produced by `tools/export_base.py`, which strips out the frozen DINOv2 backbone and the
NSD voxel embeddings. That is why they are small. The backbone is stock `dinov2_vitl14_reg` and is
downloaded separately by torch.hub, so it never needs to be in here (verified: all 344 backbone
tensors are bit-identical to the public release).

| file | size | what it is | who needs it |
|---|---|---|---|
| `ube_base_infonce.pt` | 9.6 MB | base pretrained with the contrastive (InfoNCE + entropy) objective. **This is the default and the one people should use.** | everyone |
| `ube_base_recon.pt` | 9.6 MB | same architecture and data, pretrained with reconstruction loss only. The matched control for the contrastive comparison. | anyone reproducing the contrastive-vs-reconstruction result |
| `ube_base_original.pt` | 9.6 MB | the earlier base checkpoint this work started from | completeness |
| `nsd_voxel_embed_infonce.pt` | 324 MB | the learned voxel embeddings for all 315,997 NSD voxels across the 8 NSD subjects. Only used for warm-starting a new subject's voxel embeddings instead of random init. | optional, warm start only |

Optional convenience upload:

| file | size | why |
|---|---|---|
| `dinov2_vitl14_reg4_pretrain.pth` | 1.2 GB | Meta's DINOv2 weights (Apache 2.0, freely redistributable). Only worth mirroring because many clusters block internet access on compute nodes, so `torch.hub` cannot download at runtime. People can also fetch it themselves on a login node. |

Also add a short model card (the README on the HF repo). Say: what the encoder does, that the
backbone is stock DINOv2, that the base was pretrained on NSD, that weights are for research use,
and link the repo. If you want, paste in the "how is this different from the Irani repo" section.

## 2. do NOT upload these

| file | size | why not |
|---|---|---|
| `fmri_v2.npz` | 5.8 GB | NSD betas. NSD has a data use agreement that people have to sign themselves. Redistributing the betas, even z-scored and repeat-averaged, is not ours to do. |
| `all_images_v2_224.npy` | 11 GB | the 73k NSD stimulus images (COCO). Same reasoning, and it is COCO's to distribute. |
| `sub-06_roi_vox_all_sessions.pkl`, `sub-07_...` | 0.6 GB each | human subject data from our own study. Not public, and not ours to release. |
| any fitted subject encoder (`sub-06_infonce.pth` etc.) | 1.2 GB each | these contain voxel embeddings fit to an individual's brain. Keep them internal. |

**Model weights: cleared.** You have the authors' agreement to put these up, so the three base
files and the voxel embeddings are fine to upload. Worth adding a line to the model card crediting
Beliy et al. and linking their paper, since the architecture and base training are theirs.

Note that their agreement covers *their* model. It does not cover the two things in section 2 that
belong to other people: NSD has its own data use agreement, and the subject data is ours but is
human subjects data. Those stay off HuggingFace regardless.

## 3. what someone needs to pretrain the base themselves

They do not need this to run a 2AFC test. This is only for someone who wants to redo the
pretraining (for example to retrain the contrastive objective, or to add subjects). It is a
multi-day job on several GPUs.

**The data.** NSD, from https://naturalscenesdataset.org (they sign the agreement and download it
themselves). Specifically:

- fMRI: the **`betas_fithrf_GLMdenoise_RR`** betas in **fsaverage** surface space, all 8 subjects,
  all 40 sessions each.
- Voxel selection: the **Algonauts** challenge mask (this is what defines the 315,997 voxels).
- Preprocessing, in this order: **z-score each voxel within each session**, then **average the
  repeats of each image**. One pattern per image per subject.
- Stimuli: the 73,000 NSD images (COCO), resized to 224x224 RGB.

**The file layout the training script expects** (`extras/train_base.py`, same as the Irani code):

```
data/nsd_data/
  fmri_v2.npz
    single_sub_fmri   float32 (n_images_seen_by_one_subject, n_voxels_of_that_subject)
    single_sub        int     which subject each row of single_sub_fmri belongs to
    multi_sub_fmri    float32 (n_shared_images, 315997)   voxels concatenated across all 8 subjects
    type_sample       int     one entry per NSD image: 1 = seen by a single subject,
                              2 = seen by all 8 (the shared1000), 0 = seen by nobody
    val_single_ind    int     row indices held out for validation
  num_voxels_all_subjects.npy   int (8, 2)   voxels per hemisphere per subject
  all_images_v2_224.npy         uint8 (73000, 224, 224, 3)   indexed by NSD id
```

Optionally, COCO 2017 unlabeled images as extra synthetic-fMRI training data
(`ext_images_*.npy`), which is what the `--EXT` flag in the Irani code uses.

**Hardware.** 1 GPU works but is slow. 30 epochs over the NSD training set. Budget days, not hours.

**What our contrastive change is.** `extras/train_base.py --objective infonce` adds an InfoNCE term
plus a small entropy regularizer on top of the reconstruction loss, so that a predicted pattern has
to match its own measured pattern better than the other patterns in the batch. That is what makes
the encoder better at telling near-identical images apart. `--objective recon` reproduces the
plain reconstruction baseline. Those two switches are what produced `ube_base_infonce.pt` and
`ube_base_recon.pt`.
