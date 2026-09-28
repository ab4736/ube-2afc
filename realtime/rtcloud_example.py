import os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from ube.online import PairmateScorer

# template for scoring trials during a scan, for neurofeedback
#
# the shape of this is: before the scan you cache one predicted pattern per candidate image,
# during the scan you turn the volumes collected so far into one beta per trial and hand it to
# the scorer. the scorer is pure numpy and takes microseconds, so it is never the bottleneck
#
# the only part that depends on your setup is get_trial_beta below. every real time pipeline
# already has something that does this, so plug yours in

CACHE = "cache/sub-08_preds.npz"          # made by make_pred_cache.py before the scan
TR = 1.5                                  # seconds per volume
CUT_TRS = 6                               # score at 6 TRs after onset, about 9 s. see note below


def get_trial_beta(volumes, onset_tr, mask, n_trs=CUT_TRS):
    """turn the volumes collected so far into one pattern for the current trial

    volumes   list of 3d arrays, motion corrected and on the same grid as mask
    onset_tr  which volume the trial started on
    mask      boolean 3d array picking the voxels the encoder was fit on, in the same order

    replace this with whatever your pipeline does. two common options:

    simple, no glm: average the volumes over the window and apply the mask. fast but noisier.
    proper: fit a causal glm using only the volumes up to onset+n_trs, with a boxcar for this
    trial, the other trials as nuisance regressors, drift terms and a canonical hrf, then take
    this trial's contrast. nilearn.glm.first_level does this. that is what we used and it was
    clearly better than averaging (averaging came out near chance on our data).
    """
    win = volumes[onset_tr:onset_tr + n_trs]
    if not win:
        raise ValueError("no volumes yet for this trial")
    return np.stack([v[mask] for v in win]).mean(0)          # placeholder, see docstring


def main():
    scorer = PairmateScorer.from_cache(CACHE)
    mask = np.load("mask.npy").astype(bool)                  # same voxels, same order, as training
    trials = [("pair_3_1.jpg", "pair_3_2.jpg", 10),          # shown, pairmate, onset TR
              ("pair_4_2.jpg", "pair_4_1.jpg", 30)]
    volumes = []                                             # your pipeline appends to this each TR

    for shown, foil, onset in trials:
        while len(volumes) < onset + CUT_TRS:
            volumes.append(wait_for_next_volume())           # your scanner loop
        beta = get_trial_beta(volumes, onset, mask)

        scorer.update_running_stats(beta)                    # keep the running mean and sd
        b = scorer.causal_z(beta)                            # normalise using only past trials
        out = scorer.score(b, shown, foil, z=False)

        # cpd is the number to send back as feedback. +1 means the pattern sits right on the
        # shown image's prediction, 0 is halfway to the pairmate
        print(f"trial {shown}: cpd {out['cpd']:+.2f}  correct {out['correct']}")
        send_feedback(out["cpd"])                            # your display


# two things we measured that matter for timing
#
# 1. score at about 6 to 9 s after onset. accuracy peaked at 9 s and waiting longer made it
#    worse, because you start averaging in the falling part of the response
# 2. do not widen the window to decode earlier. a wide window has to include pre response
#    volumes, which delays it. narrow windows crossed 0.9 sooner


def wait_for_next_volume():
    raise NotImplementedError("plug in your scanner or rt-cloud volume source")


def send_feedback(cpd):
    raise NotImplementedError("plug in your display")


if __name__ == "__main__":
    main()
