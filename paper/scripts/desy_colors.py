"""
The DESY colour palette.

Convention used throughout the figures:
    orange     the binaryfog background
    cyan       what it is compared against (the (A, gamma) power law, the flat-prior posterior)
    petrol     the new physics alone
    gruen      the binary background on its own
    dunkelrot  the measured free spectrum, drawn as violins
"""

DESYcyan = (0 / 255, 159 / 255, 223 / 255)
DESYorange = (241 / 255, 143 / 255, 31 / 255)
DESYrot = (230 / 255, 65 / 255, 35 / 255)
DESYdunkelrot = (153 / 255, 0 / 255, 0 / 255)
DESYdunkelblau = (0 / 255, 74 / 255, 110 / 255)
DESYpetrol = (0 / 255, 106 / 255, 133 / 255)
DESYtuerkis = (45 / 255, 164 / 255, 164 / 255)
DESYgruen = (0 / 255, 168 / 255, 62 / 255)
DESYhellgruen = (122 / 255, 158 / 255, 31 / 255)
DESYgelb = (230 / 255, 169 / 255, 15 / 255)
DESYaubergine = (139 / 255, 60 / 255, 126 / 255)
DESYviolett = (146 / 255, 124 / 255, 184 / 255)
DESYlila = (82 / 255, 78 / 255, 156 / 255)
DESYmagenta = (208 / 255, 0 / 255, 111 / 255)
DESYhellbraun = (170 / 255, 132 / 255, 106 / 255)
DESYgrau = (0.45, 0.45, 0.45)


def desy_ramp(n):
    """A sequential ramp in the DESY palette, for showing one parameter moved across its range.

    Runs dark blue to petrol to turquoise to green to yellow, which stays ordered in greyscale.
    """
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    stops = [DESYdunkelblau, DESYpetrol, DESYtuerkis, DESYhellgruen, DESYgelb]
    return LinearSegmentedColormap.from_list("desy", stops)(np.linspace(0.0, 1.0, n))
