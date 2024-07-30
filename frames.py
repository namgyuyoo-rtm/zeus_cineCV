import argparse
from pathlib import Path
from datetime import datetime, timedelta
import csv
import math

import cv2
import numpy as np
from tqdm import tqdm
from pycine.raw import read_frames
from pycine.file import read_header

def apply_gamma(rgb_image, setup):
    # FIXME: using 2.2 for now because 8.0 from the sample image seems way out of place
    # --> this is not at all how vri is doing it!
    rgb_image = np.abs(rgb_image) ** (1.0 / 2.2)
    # rgb_image[:, :, 0] **= (1.0 / (setup.fGammaR + setup.fGamma))
    # rgb_image[:, :, 1] **= (1.0 / setup.fGamma)
    # rgb_image[:, :, 2] **= (1.0 / (setup.fGammaB + setup.fGamma))

    return rgb_image


def whitebalance_raw(raw, setup, pattern):
    cmCalib = np.asarray(setup.cmCalib).reshape(3, 3)
    whitebalance = np.diag(cmCalib)
    whitebalance = [1.193739671606806, 1.0, 1.7885392465247287]

    # FIXME: maybe use .copy()
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
    """Order from:
    http://www.visionresearch.com/phantomzone/viewtopic.php?f=20&t=572#p3884
    """
    # 1. Offset the raw image by the amount in flare

    # 2. White balance the raw picture using the white balance component of cmatrix
    BayerPatterns = {3: "gbrg", 4: "rggb"}
    pattern = BayerPatterns[setup.CFA]

    raw = whitebalance_raw(raw.astype(np.float32), setup, pattern).astype(np.uint16)

    # 3. Debayer the image
    rgb_image = cv2.cvtColor(raw, cv2.COLOR_BAYER_GB2RGB)

    # convert to float
    rgb_image = rgb_image.astype(np.float32) / (2**bpp - 1)

    # return rgb_image

    # 4. Apply the color correction matrix component of cmatrix
    #
    # From the documentation:
    # ...should decompose this
    # matrix in two components: a diagonal one with the white balance to be
    # applied before interpolation and a normalized one to be applied after
    # interpolation.

    cmCalib = np.asarray(setup.cmCalib).reshape(3, 3)

    # normalize matrix
    ccm = cmCalib / cmCalib.sum(axis=1)[:, np.newaxis]

    # or should it be normalized this way?
    ccm2 = cmCalib.copy()
    ccm2[0][0] = 1 - ccm2[0][1] - ccm2[0][2]
    ccm2[1][1] = 1 - ccm2[1][0] - ccm2[1][2]
    ccm2[2][2] = 1 - ccm2[2][0] - ccm2[2][1]

    m = np.asarray(
        [
            1.4956012040024347,
            -0.5162879962189262,
            0.020686792216491584,
            -0.09884672458400766,
            0.757682383759598,
            0.34116434082440983,
            -0.04121405804689133,
            -0.5527871476076358,
            1.5940012056545272,
        ]
    ).reshape(3, 3)

    rgb_image = np.dot(rgb_image, m.T)
    # rgb_reshaped = rgb_image.reshape((rgb_image.shape[0] * rgb_image.shape[1], rgb_image.shape[2]))
    # rgb_image = np.dot(m, rgb_reshaped.T).T.reshape(rgb_image.shape)

    # 5. Apply the user RGB matrix umatrix
    # cmUser = np.asarray(setup.cmUser).reshape(3, 3)
    # rgb_image = np.dot(rgb_image, cmUser.T)

    # 6. Offset the image by the amount in offset

    # 7. Apply the global gain

    # 8. Apply the per-component gains red, green, blue

    # 9. Apply the gamma curves; the green channel uses gamma, red uses gamma + rgamma and blue uses gamma + bgamma
    rgb_image = apply_gamma(rgb_image, setup)

    # 10. Apply the tone curve to each of the red, green, blue channels
    fTone = np.asarray(setup.fTone)

    # 11. Add the pedestals to each color channel, and linearly rescale to keep the white point the same.

    # 12. Convert to YCrCb using REC709 coefficients

    # 13. Scale the Cr and Cb components by chroma.

    # 14. Rotate the Cr and Cb components around the origin in the CrCb plane by hue degrees.

    return rgb_image


def to_3ch_gray(frame: np.ndarray):
    return cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)[
        ..., None
    ].repeat(3, axis=-1)


def get_frames(
    video_path: str, start_frame: int, count: int, stride: int = 1, cfa: int = 3
):
    raw_images, setup, bpp = read_frames(
        video_path, start_frame=start_frame, count=count
    )
    setup.CFA = cfa

    for i, raw_image in enumerate(raw_images):
        if i % stride == 0:
            yield to_3ch_gray(color_pipeline(raw_image, setup=setup, bpp=bpp))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .cine files and extract frames.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input .cine file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory")
    args = parser.parse_args()

    video_path = Path(args.input)
    save_dir = Path(args.output)

    start_frame = 0
    frame_count = None
    stride = 10
    group_interval = timedelta(milliseconds=300)

    header = read_header(video_path)
    trigger_time = header["cinefileheader"].TriggerTime
    total_frames = header["cinefileheader"].ImageCount
    frame_rate = header["setup"].FrameRate
    assert start_frame < total_frames, "Start frame is out of range"
    assert (start_frame + (frame_count or 0) < total_frames), "Frame count is out of range"

    print(f"Total Frames: {total_frames}, Frame Rate: {frame_rate}")

    start_time = datetime.fromtimestamp(trigger_time.seconds + trigger_time.fractions / 1e6)
    end_time = start_time + timedelta(seconds=total_frames / frame_rate)
    time_per_frame = timedelta(seconds=1 / frame_rate)

    print(f"Start Time: {start_time}, End Time: {end_time}, Time per Frame: {time_per_frame}")

    total_duration = end_time - start_time
    total_groups = math.ceil(total_duration / group_interval)

    frames = get_frames(video_path, start_frame, frame_count, stride)
    pbar = tqdm(frames, total=np.ceil((frame_count or total_frames) / stride), desc="Processing Frames")

    max_digits = len(str(total_frames))
    time_format = "%y%m%d_%H%M%S.%f"

    csv_file = save_dir / f"{video_path.stem}_processing_log.csv"
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Frame Number", "Timestamp", "Group", "Filename"])

        current_group = 1
        group_start_time = start_time

        for i, frame in enumerate(pbar):
            frame_time = start_time + i * stride * time_per_frame
            
            if frame_time - group_start_time >= group_interval:
                current_group += 1
                group_start_time = frame_time
            
            group_dir = save_dir / video_path.stem / f"frame_group_{current_group:04d}of{total_groups:04d}"
            group_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{i*stride:0{max_digits}d}_{video_path.stem}_{frame_time.strftime(time_format)}.png"
            save_path = group_dir / filename

            cv2.imwrite(save_path.as_posix(), frame)

            csv_writer.writerow([i*stride, frame_time.strftime(time_format), current_group, filename])

    print(f"Processing complete. Log file saved to {csv_file}")