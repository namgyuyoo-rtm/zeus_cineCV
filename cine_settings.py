# cine_settings.py

import cv2
import numpy as np

def apply_gamma(rgb_image, setup):
    rgb_image = np.abs(rgb_image) ** (1.0 / 2.2)
    return rgb_image

def whitebalance_raw(raw, setup, pattern):
    cmCalib = np.asarray(setup.cmCalib).reshape(3, 3)
    whitebalance = np.diag(cmCalib)
    whitebalance = [1.193739671606806, 1.0, 1.7885392465247287]

    wb_raw = np.ma.MaskedArray(raw)

    wb_raw.mask = gen_mask(pattern, "r", wb_raw)
    wb_raw *= whitebalance[0]
    wb_raw.mask = gen_mask(pattern, "g", wb_raw)
    wb_raw *= whitebalance[1]
    wb_raw.mask = gen_mask(pattern, "b", wb_raw)
    wb_raw *= whitebalance[2]

    wb_raw.mask = np.ma.nomask

    return wb_raw

def gen_mask(pattern, c, image):
    def color_kern(pattern, c):
        return np.array(
            [[pattern[0] != c, pattern[1] != c], [pattern[2] != c, pattern[3] != c]]
        )

    (h, w) = image.shape[:2]
    cells = np.ones((h // 2, w // 2))

    return np.kron(cells, color_kern(pattern, c))

def color_pipeline(raw, setup, bpp=12):
    BayerPatterns = {3: "gbrg", 4: "rggb"}
    pattern = BayerPatterns[setup.CFA]

    raw = whitebalance_raw(raw.astype(np.float32), setup, pattern).astype(np.uint16)
    rgb_image = cv2.cvtColor(raw, cv2.COLOR_BAYER_GB2RGB)
    rgb_image = rgb_image.astype(np.float32) / (2**bpp - 1)

    cmCalib = np.asarray(setup.cmCalib).reshape(3, 3)
    m = np.asarray([
        1.4956012040024347, -0.5162879962189262, 0.020686792216491584,
        -0.09884672458400766, 0.757682383759598, 0.34116434082440983,
        -0.04121405804689133, -0.5527871476076358, 1.5940012056545272
    ]).reshape(3, 3)

    rgb_image = np.dot(rgb_image, m.T)
    rgb_image = apply_gamma(rgb_image, setup)

    return rgb_image