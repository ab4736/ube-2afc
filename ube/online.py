import numpy as np

# scoring one trial as it happens, for neurofeedback
#
# the expensive part (running the encoder on the candidate images) is done ahead of time and
# saved by ube/cache.py. at trial time all that is left is two correlations and a projection,
# which is microseconds, so this can sit inside a real time loop
#
# there is no gpu and no torch in here on purpose, so it can run wherever your scanner
# pipeline runs


def _z(a):
    a = np.asarray(a, np.float64)
    a = a - a.mean()
    return a / (a.std() + 1e-8)


class PairmateScorer:
    """holds the precomputed predictions and scores trials one at a time

    preds  (n_images, n_voxels) predicted patterns
    names  image ids, same order as preds
    """

    def __init__(self, preds, names):
        self.preds = np.asarray(preds, np.float32)
        self.names = [str(n) for n in names]
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.zp = np.stack([_z(p) for p in self.preds])        # z-score once, reuse every trial
        self._sum = np.zeros(self.preds.shape[1])              # running stats for causal z
        self._sumsq = np.zeros(self.preds.shape[1])
        self._n = 0

    @classmethod
    def from_cache(cls, path):
        from ube.cache import load_pred_cache
        P, names = load_pred_cache(path)
        return cls(P, names)

    def update_running_stats(self, beta):
        """feed each trial's raw beta in as it arrives, so causal_z can normalise honestly"""
        b = np.asarray(beta, np.float64)
        self._sum += b
        self._sumsq += b * b
        self._n += 1

    def causal_z(self, beta):
        """z-score using only the trials seen so far, which is all you have online

        offline you would z-score against the whole session, but online that would peek at the
        future. with fewer than 2 trials there is nothing to normalise against, so this just
        z-scores the pattern across voxels instead
        """
        b = np.asarray(beta, np.float64)
        if self._n < 2:
            return _z(b)
        mean = self._sum / self._n
        var = np.maximum(self._sumsq / self._n - mean ** 2, 1e-12)
        return (b - mean) / np.sqrt(var)

    def score(self, beta, shown, foil, z=True):
        """score one trial

        beta   measured pattern for this trial, length n_voxels
        shown  image id that was actually presented
        foil   its pairmate
        z      z-score the beta across voxels first (leave True unless you already did)

        returns r_shown, r_foil, cpd, correct, margin
          cpd  +1 means the measured pattern sits on the shown image's prediction,
               0 is halfway between the two, -1 is on the pairmate's
        """
        for n in (shown, foil):
            if n not in self.idx:
                raise KeyError(f"'{n}' is not in the prediction cache. it has {len(self.names)} "
                               f"images, first few: {self.names[:3]}")
        i, j = self.idx[shown], self.idx[foil]
        b = _z(beta) if z else np.asarray(beta, np.float64)
        n_vox = self.preds.shape[1]
        if b.shape[0] != n_vox:
            raise ValueError(f"beta has {b.shape[0]} voxels but the predictions have {n_vox}")

        r_shown = float((b * self.zp[i]).mean())               # pearson, patterns already z-scored
        r_foil = float((b * self.zp[j]).mean())

        axis = self.preds[i] - self.preds[j]                   # the pairmate axis
        mid = (self.preds[i] + self.preds[j]) / 2.0
        raw = np.asarray(beta, np.float64)
        cpd = 2.0 * float((raw - mid) @ axis) / (float(axis @ axis) + 1e-12)

        return dict(shown=shown, foil=foil, r_shown=r_shown, r_foil=r_foil,
                    margin=r_shown - r_foil, correct=bool(r_shown > r_foil), cpd=cpd)

    def pick(self, beta, a, b_img, z=True):
        """when you do not know which image was shown, just say which one the brain looks like"""
        out = self.score(beta, a, b_img, z=z)
        return a if out["r_shown"] > out["r_foil"] else b_img, out
