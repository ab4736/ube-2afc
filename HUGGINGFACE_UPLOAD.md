# what to put on hugging face

This is for you (Akash), not for the people using the repo. It lists what to upload so someone can
run everything with one download command and no internet access on their compute nodes.

Make a **model** repo at `huggingface.co/new`, name it `ube-2afc`. `download_checkpoints.sh` points
at `ab4736/ube-2afc` and expects these exact filenames, so if you name it something else, change
the `REPO=` line in that script or have people set `UBE_HF_REPO`.

## 1. the encoder weights

Produced by `tools/export_base.py`, which strips out the frozen DINOv2 backbone and the NSD voxel
embeddings, which is why they are tiny. They are plain tensors plus a config dict, so they load
with `weights_only=True` and run no pickled code.

Already made, sitting in `/scratch/gpfs/KNORMAN/ab4736/ube-2afc/ckpt_export/`:

| file | size | what it is |
|---|---|---|
| `ube_base_infonce.pt` | 9.6 MB | base pretrained with the contrastive objective. the default, the one people should use |
| `ube_base_recon.pt` | 9.6 MB | same architecture and data, reconstruction loss only. the matched control |
| `ube_base_original.pt` | 9.6 MB | the earlier base this work started from |
| `nsd_voxel_embed_infonce.pt` | 324 MB | learned voxel embeddings for all 315,997 NSD voxels. only needed for warm starting |

## 2. the DINOv2 backbone

| file | size |
|---|---|
| `dinov2_vitl14_reg4_pretrain.pth` | 1.2 GB |

Upload this. It is Meta's, Apache 2.0, so redistributing it is fine, and it matters more than it
looks. Without it, people depend on `torch.hub` reaching the internet, and most clusters block that
on compute nodes. With it in the download, `ube/load.py` builds the backbone from the copy of the
DINOv2 source vendored in the repo and never touches the network. That path is tested and gives
identical results.

Copy it from:

```
/scratch/gpfs/KNORMAN/ab4736/brainit-fmri/data/external_models/torch_hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth
```

## 3. the NSD training data

Only needed by someone who wants to pretrain a base from scratch. Put it in a `nsd_data/` folder
inside the same repo, which is where `WITH_NSD=1 ./download_checkpoints.sh` looks.

| file | size | what it is |
|---|---|---|
| `nsd_data/fmri_v2.npz` | 5.8 GB | NSD betas, already preprocessed: `betas_fithrf_GLMdenoise_RR` in fsaverage, Algonauts mask, z-scored within session, repeats averaged |
| `nsd_data/num_voxels_all_subjects.npy` | 256 B | voxels per hemisphere per subject |
| `nsd_data/all_images_v2_224.npy` | 11 GB | the 73k NSD stimulus images at 224x224, indexed by NSD id |

Copy them from `/scratch/gpfs/KNORMAN/qanguyen/Brain-IT/data/nsd_data/`.

**One decision before the repo goes public.** The Irani authors cleared their model, but NSD is not
theirs to license. NSD has its own terms and a data access form, and people are formally supposed
to agree to those themselves. Other groups do host preprocessed NSD publicly (MedARC does for
MindEye), so this is normal practice, but the safe version is one of:

- set the HF repo to **gated**, so people accept terms before downloading, or
- keep it private and add people individually, or
- at minimum say in the model card that the data is NSD, link https://naturalscenesdataset.org,
  and note that users are expected to have agreed to NSD's terms.

Any of those is fine. Doing none of them is the only option I would avoid.

## 4. how to upload

The browser uploader will not handle the 11 GB file, so use the CLI:

```bash
pip install huggingface_hub
huggingface-cli login

# the small weights
cd /scratch/gpfs/KNORMAN/ab4736/ube-2afc/ckpt_export
huggingface-cli upload ab4736/ube-2afc . . --repo-type=model

# the backbone
cp /scratch/gpfs/KNORMAN/ab4736/brainit-fmri/data/external_models/torch_hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth .
huggingface-cli upload ab4736/ube-2afc dinov2_vitl14_reg4_pretrain.pth --repo-type=model

# the nsd data into a nsd_data/ folder in the same repo. 17 GB, slow
huggingface-cli upload ab4736/ube-2afc \
    /scratch/gpfs/KNORMAN/qanguyen/Brain-IT/data/nsd_data nsd_data --repo-type=model
```

## 5. do not upload

| what | why |
|---|---|
| our own subjects' beta files | human subject data from our own study, not ours to release |
| any encoder fitted to one of our subjects | those voxel embeddings are fit to an individual's brain |

## 6. model card

A short README on the HF repo covering: what the encoder does, that the backbone is stock DINOv2,
that the base was pretrained on NSD, credit and links to Beliy et al. and
https://github.com/WeizmannVision/brainit-fmri, the NSD link and terms note from section 3, and a
link back to the code repo.

## 7. check it worked

Run this in an empty folder. It is the exact path a new user hits first:

```bash
git clone https://github.com/ab4736/ube-2afc.git && cd ube-2afc
./download_checkpoints.sh
ls -la checkpoints        # three 9.6 MB .pt files and the 1.2 GB backbone
```
