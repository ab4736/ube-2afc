import os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from ube.online import PairmateScorer

# SKELETON, NOT RUNNABLE AS IS. this file shows how to call the scorer from inside a real time
# loop. it does not integrate with rt-cloud or any other acquisition system, and three functions
# below are left unimplemented on purpose because they depend entirely on your setup:
# get_trial_beta, wait_for_next_volume and send_feedback. running this file as written will stop
# at the first of those.
#
# what it is actually showing you is the order of the calls, which is: before the scan you cache
# one predicted pattern per candidate image, and during the scan you turn the volumes collected
# so far into one response estimate per trial and pass it to the scorer. the scoring itself is
# pure numpy and takes microseconds, so it is never the part that limits you.

CACHE = "cache/sub-08_preds.npz"          # made by make_pred_cache.py before the scan
TR = 1.5                                  # seconds per volume
CUT_TRS = 6                               # score at 6 TRs after onset, about 9 s. see note below


def get_trial_beta(volumes, onset_tr, mask, n_trs=CUT_TRS):
    """turn the volumes collected so far into one pattern for the current trial

    volumes   list of 3d arrays, motion corrected and on the same grid as mask
    onset_tr  which volume the trial started on
    mask      boolean 3d array picking the voxels the encoder was fit on, in the same order

    you have to implement this, because it is the part that differs between setups. what we
    used, and what we recommend, is a causal glm: fit using only the volumes up to onset+n_trs,
    with a boxcar regressor for this trial, the other trials as nuisance regressors, drift terms
    and a canonical hrf, then take this trial's contrast as the response estimate.
    nilearn.glm.first_level provides the pieces for this.

    the obvious alternative, averaging the volumes across the window and applying the mask, is
    much simpler but performed close to chance on our data, so we do not provide it here as a
    default that could be copied by accident.

    whatever you use, the vector it returns has to cover the same voxels, in the same order, as
    the betas the encoder was fitted on.
    """
    raise NotImplementedError(
        "implement get_trial_beta for your acquisition setup, see the docstring above")


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
#    volumes, which delays it. narrow windows got there sooner


def wait_for_next_volume():
    raise NotImplementedError("plug in your scanner or rt-cloud volume source")


def send_feedback(cpd):
    raise NotImplementedError("plug in your display")


if __name__ == "__main__":
    main()
