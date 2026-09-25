"""
Download the free-spectrum data products, so that a user needs nothing but this package.

    ng15_cp_v1, ng15_hd_v1, ipta2_cp_v1   PTArcade data, Zenodo 10495907 (2024), CC-BY-4.0
    ng15_hdall_v1, ng15_hd_dmgp_v1        NANOGrav, Zenodo 8060824 (2023, v1.0.0), CC-BY-4.0: the
                                          Hellings-Downs spectrum inferred together with a
                                          monopole, a dipole and an uncorrelated common process,
                                          and the one modelling dispersion measure variations as a
                                          Gaussian process rather than as piecewise offsets
    ng15_cp_v2, ng15_hd_v2, ng15_hdall_v2 NANOGrav, Zenodo 21844115 (7 August 2026), CC-BY-4.0:
                                          the NANOGrav 15 yr free spectra re-run after the
                                          parallel-tempering bug in PTMCMCSampler, see the erratum
                                          Astrophys. J. Lett. 1006 (2026) L67. That record does not
                                          carry a corrected Gaussian-process product, so
                                          ng15_hd_dmgp_v1 is the 2024 one.

Every URL below points at a specific Zenodo version rather than at the concept record, so a later
release by NANOGrav cannot silently change what this downloads.

The 2024 products carry a grid of log10 rho from -15.1 to -0.9, but the Markov chains never visited
values below about -9; there the stored density is a numerical floor with isolated single-sample
spikes, which a profile likelihood can find. The 2026 products keep only the sampled range.

    python -m binaryfog.fetch            downloads everything into ~/.binaryfog/freespec
"""
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

OLD = "https://zenodo.org/records/10495907/files/{}.zip?download=1"
# v1.0.0 rather than the later v1.1.0: this is the release the published figures were made with,
# and the two differ in the stored grid and density
NG23 = "https://zenodo.org/records/8060824/files/NANOGrav15yr_KDE-FreeSpectra_v1.0.0.zip?download=1"
NEW = "https://zenodo.org/records/21844115/files/ceffyl_data.zip?download=1"
MEMBERS = {
    "ng15_cp_v1": (OLD, "ng15_30f_fs{cp}_ceffyl"),
    "ng15_hd_v1": (OLD, "ng15_30f_fs{hd}_ceffyl"),
    "ipta2_cp_v1": (OLD, "ipta2_30f_fs{cp}_ceffyl"),
    "ng15_hdall_v1": (NG23, "30f_fs{hd+mp+dp+cp}_ceffyl_hd-only"),
    "ng15_hd_dmgp_v1": (NG23, "30f_fs{hd}_100fDMGP_ceffyl"),
    # the directory name of the Hellings-Downs product in the 2026 archive ends in a space, which
    # _members below tolerates, so that the entry here reads as the name was meant
    "ng15_cp_v2": (NEW, "ceffyl_data/30f_fs{cp}_ceffyl"),
    "ng15_hd_v2": (NEW, "ceffyl_data/30f_fs{hd}_ceffyl"),
    "ng15_hdall_v2": (NEW, "ceffyl_data/30f_fs{hd+mp+dp+cp}_ceffyl_hd-only"),
}
NAMES = list(MEMBERS)


def _members(zf, prefix):
    """The files under `prefix`, matching its directories up to surrounding whitespace.

    Some directory names in the published archives carry a trailing space. Comparing the parts of a
    path with that stripped means a name that is later tidied up keeps working, and a name that is
    not does too.
    """
    want = [q.strip() for q in prefix.split("/")]
    for m in zf.namelist():
        if m.endswith("/"):
            continue
        parts = [q.strip() for q in m.split("/")]
        if parts[:len(want)] == want and len(parts) > len(want):
            yield m


def fetch(name, root=None, timeout=600):
    """Download one data set into root/name. Returns its path."""
    from .freespec import DEFAULT_DIR
    root = Path(root) if root is not None else DEFAULT_DIR
    if name not in MEMBERS:
        raise ValueError(f"unknown data set {name!r}; available: {', '.join(NAMES)}")
    url, prefix = MEMBERS[name]
    if url is OLD:
        url = OLD.format(prefix.replace("{", "%7B").replace("}", "%7D"))
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        zf = zipfile.ZipFile(io.BytesIO(r.read()))
    found = False
    for m in _members(zf, prefix):
        if True:
            with zf.open(m) as src, open(out / Path(m).name, "wb") as dst:
                shutil.copyfileobj(src, dst)
            found = True
    if not found or not (out / "density.npy").is_file():
        raise RuntimeError(f"{name}: nothing extracted from {prefix!r}; the archive may have moved")
    return out


if __name__ == "__main__":
    for n in NAMES:
        print(f"  {n:15s} -> {fetch(n)}")
