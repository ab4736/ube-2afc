# what is on hugging face

This file is a record for me rather than documentation for the people using the repository. It
describes what has been uploaded to HuggingFace and why, so that the contents of the model
repository can be reconstructed or corrected later if anything needs to change.

The repository is a model repository at `ab4736/ube-2afc`. The download script in the code
repository points at that name and expects the filenames listed below, so if the repository is ever
renamed, either the `REPO=` line in `download_checkpoints.sh` needs to be updated or users need to
set `UBE_HF_REPO` to the new name.

## 1. the encoder weights

These files were produced by `tools/export_base.py`, which removes the frozen DINOv2 backbone and
the NSD voxel embeddings from the original checkpoints. That is why they are small, since almost
all of the size of the original files came from those two components rather than from the encoder
itself. Each file contains plain tensors together with a configuration dictionary, which means it
can be loaded with `weights_only=True` and executes no pickled code when opened.

The local copies are in `/scratch/gpfs/KNORMAN/ab4736/ube-2afc/ckpt_export/`:

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

This is Meta's DINOv2 checkpoint, released under Apache 2.0, so redistributing it is permitted. It
matters more than its role might suggest, because without a local copy the code depends on
`torch.hub` being able to reach the internet, and compute nodes on most clusters cannot. When the
file is present, `ube/load.py` constructs the backbone from the copy of the DINOv2 source that is
vendored in the code repository and loads these weights into it, so no network access is required
at any point. That path has been tested and produces identical results to the `torch.hub` route.

The copy that was uploaded came from
`/scratch/gpfs/KNORMAN/ab4736/brainit-fmri/data/external_models/torch_hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth`,
and the downloaded file was confirmed to match it exactly.

## 3. the NSD training data

This data is only needed by someone who intends to pretrain a base encoder from scratch, and it is
stored in a `nsd_data/` folder inside the same model repository, which is where
`WITH_NSD=1 ./download_checkpoints.sh` retrieves it from.

| file | size | what it is |
|---|---|---|
| `nsd_data/fmri_v2.npz` | 5.8 GB | NSD betas, already preprocessed: `betas_fithrf_GLMdenoise_RR` in fsaverage, Algonauts mask, z-scored within session, repeats averaged |
| `nsd_data/num_voxels_all_subjects.npy` | 256 B | voxels per hemisphere per subject |
| `nsd_data/all_images_v2_224.npy` | 11 GB | the 73k NSD stimulus images at 224x224, indexed by NSD id |

These files came from `/scratch/gpfs/KNORMAN/qanguyen/Brain-IT/data/nsd_data/`.

One point is worth recording about the NSD data specifically. The Irani lab authors gave permission
for their model to be distributed, but NSD is not theirs to license. NSD is released under its own
terms and has a data access form, and users are formally expected to agree to those terms
themselves. Other groups do host preprocessed NSD data publicly, as MedARC does for MindEye, so
distributing it in this form is established practice rather than unusual, but the options that keep
that clearly in order are as follows:

- set the HF repo to **gated**, so people accept terms before downloading, or
- keep it private and add people individually, or
- at minimum say in the model card that the data is NSD, link https://naturalscenesdataset.org,
  and note that users are expected to have agreed to NSD's terms.

The repository is currently public and ungated, and the model card credits NSD, links to
naturalscenesdataset.org and states that users are expected to follow NSD's terms and cite the
original paper. If a stronger form of consent is wanted later, gating the repository is a setting
on HuggingFace and requires no change to the code.

## 4. how the upload was done

The browser uploader cannot handle files of this size, so the command line interface was used
instead. These are the commands that produced the current contents of the repository, and they are
recorded here so that individual files can be replaced later without having to work out the syntax
again.

```bash
pip install huggingface_hub
huggingface-cli login

# the encoder weights and the model card
cd /scratch/gpfs/KNORMAN/ab4736/ube-2afc/ckpt_export
hf upload ab4736/ube-2afc . . --repo-type=model

# the backbone
B=/scratch/gpfs/KNORMAN/ab4736/brainit-fmri/data/external_models/torch_hub/checkpoints
hf upload ab4736/ube-2afc $B/dinov2_vitl14_reg4_pretrain.pth \
    dinov2_vitl14_reg4_pretrain.pth --repo-type=model

# the nsd data, one file at a time into a nsd_data/ folder in the same repository
N=/scratch/gpfs/KNORMAN/qanguyen/Brain-IT/data/nsd_data
for f in num_voxels_all_subjects.npy fmri_v2.npz all_images_v2_224.npy \
         all_images_v2_112.npy all_images_v2_256.npy; do
  hf upload ab4736/ube-2afc "$N/$f" "nsd_data/$f" --repo-type=model
done
```

The image files at 112 and 256 pixels were included in addition to the 224 pixel version that the
training code actually reads, so that anyone working from this data does not have to resize the
images themselves. The CLIP-preprocessed image array was deliberately left out, because nothing in
this repository reads it and it would have added 62 GB.

## 5. deliberately not uploaded

Two categories of file were held back, and both concern our own participants rather than the model.
Our subjects' beta files are human subject data from our own study and are not ours to release, and
any encoder that has been fitted to one of those subjects consists of voxel embeddings learned from
an individual person's brain, which makes it subject data rather than a model in any useful sense.

## 6. model card

The model card is uploaded as `README.md` in the model repository, and its source is kept at
`ckpt_export/README.md` in the code repository so that it can be edited and re-uploaded in the same
way as any other file. It describes what the encoder does, notes that the backbone is unmodified
DINOv2 and that the base was pretrained on NSD, credits Beliy et al. and links both their paper and
https://github.com/WeizmannVision/brainit-fmri, links NSD and states the expectation that users
follow their terms, and points back to the code repository.

## 7. verification

The download path was tested by running the following in an empty directory, which is the sequence
a new user encounters first. All four files arrived, the downloaded base loaded correctly, and the
backbone matched the original file byte for byte.

```bash
git clone https://github.com/ab4736/ube-2afc.git && cd ube-2afc
./download_checkpoints.sh
ls -la checkpoints        # three 9.6 MB .pt files and the 1.2 GB backbone
```
