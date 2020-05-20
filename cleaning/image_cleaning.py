import matplotlib.pyplot as plt
import numpy as np

from traitlets.config.loader import Config
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from ctapipe.image.cleaning import tailcuts_clean, mars_cleaning_1st_pass
from ctapipe.image.extractor import LocalPeakWindowSum
from ctapipe_io_lst import LSTEventSource
from lstchain.calib.camera.r0 import LSTR0Corrections
from lstchain.calib.camera.calibrator import LSTCameraCalibrator


charge_config = Config({
    "LocalPeakWindowSum":{
        "window_shift":3,
        "window_width":6
    }
})


def check_interleave_pedestal_cleaning(path_list, calib_time_file, calib_file, max_events=10000):

    signal_place_after_clean = np.zeros(1855)
    sum_ped_ev = 0
    alive_ped_ev = 0

    for path in path_list:
        print(path)
        r0_r1_calibrator = LSTR0Corrections(pedestal_path=None,
                                            r1_sample_start=3,
                                            r1_sample_end=39)

        r1_dl1_calibrator = LSTCameraCalibrator(calibration_path=calib_file,
                                                time_calibration_path=calib_time_file,
                                                extractor_product="LocalPeakWindowSum",
                                                config=charge_config,
                                                allowed_tels=[1])

        reader = LSTEventSource(input_url=path, max_events=max_events)


        for i, ev in enumerate(reader):
            r0_r1_calibrator.calibrate(ev)
            if i%10000 == 0:
                print(ev.r0.event_id)

            if ev.lst.tel[1].evt.tib_masked_trigger == 32:
                sum_ped_ev += 1

                r1_dl1_calibrator(ev)

                img = ev.dl1.tel[1].image

                geom = ev.inst.subarray.tel[1].camera
                clean = tailcuts_clean(
                                    geom,
                                    img,
                                    picture_thresh=6,
                                    boundary_thresh=3,
                                    min_number_picture_neighbors=1,
                                    keep_isolated_pixels=False
                                    )

                cleaned = img.copy()
                cleaned[~clean] = 0.0

                signal_place_after_clean[np.where(clean == True)] += 1

                if np.sum(cleaned>0) > 0:
                    alive_ped_ev += 1

    fig, ax = plt.subplots(figsize=(8, 8))
    geom = ev.inst.subarray.tel[1].camera

    disp0 = CameraDisplay(geom, ax=ax)
    disp0.image = signal_place_after_clean/sum_ped_ev
    disp0.add_colorbar(ax=ax, label="N times signal remain after cleaning")
    disp0.cmap = 'gnuplot2'
    ax.set_title("{} \n {}/{}".format(path.split("/")[-1][8:21], alive_ped_ev, sum_ped_ev))

    print(path.split("/")[-1][8:21])
    print("{}/{}".format(alive_ped_ev, sum_ped_ev))

    ax.set_xlabel(" ")
    ax.set_ylabel(" ")
    plt.tight_layout()
    plt.show()

    return signal_place_after_clean, sum_ped_ev


def my_tailcuts_clean(
    geom,
    image,
    noise_pixels_id_list,
    picture_thresh=7,
    boundary_thresh=5,
    keep_isolated_pixels=False,
    min_number_picture_neighbors=0,
):

    """Clean an image by selection pixels that pass a two-threshold
    tail-cuts procedure.  The picture and boundary thresholds are
    defined with respect to the pedestal dispersion. All pixels that
    have a signal higher than the picture threshold will be retained,
    along with all those above the boundary threshold that are
    neighbors of a picture pixel.
    To include extra neighbor rows of pixels beyond what are accepted, use the
    `ctapipe.image.dilate` function.
    Parameters
    ----------
    geom: `ctapipe.instrument.CameraGeometry`
        Camera geometry information
    image: array
        pixel values
    picture_thresh: float or array
        threshold above which all pixels are retained
    boundary_thresh: float or array
        threshold above which pixels are retained if they have a neighbor
        already above the picture_thresh
    keep_isolated_pixels: bool
        If True, pixels above the picture threshold will be included always,
        if not they are only included if a neighbor is in the picture or
        boundary
    min_number_picture_neighbors: int
        A picture pixel survives cleaning only if it has at least this number
        of picture neighbors. This has no effect in case keep_isolated_pixels is True
    Returns
    -------
    A boolean mask of *clean* pixels.  To get a zero-suppressed image and pixel
    list, use `image[mask], geom.pix_id[mask]`, or to keep the same
    image size and just set unclean pixels to 0 or similar, use
    `image[~mask] = 0`
    """
    pixels_above_picture = image >= picture_thresh
    image[noise_pixels_id_list] = False

    if keep_isolated_pixels or min_number_picture_neighbors == 0:
        pixels_in_picture = pixels_above_picture
    else:
        # Require at least min_number_picture_neighbors. Otherwise, the pixel
        #  is not selected
        number_of_neighbors_above_picture = geom.neighbor_matrix_sparse.dot(
            pixels_above_picture.view(np.byte)
        )
        pixels_in_picture = pixels_above_picture & (
            number_of_neighbors_above_picture >= min_number_picture_neighbors
        )

    # by broadcasting together pixels_in_picture (1d) with the neighbor
    # matrix (2d), we find all pixels that are above the boundary threshold
    # AND have any neighbor that is in the picture
    pixels_above_boundary = image >= boundary_thresh
    pixels_with_picture_neighbors = geom.neighbor_matrix_sparse.dot(pixels_in_picture)
    if keep_isolated_pixels:
        return (
            pixels_above_boundary & pixels_with_picture_neighbors
        ) | pixels_in_picture
    else:
        pixels_with_boundary_neighbors = geom.neighbor_matrix_sparse.dot(
            pixels_above_boundary
        )
        return (pixels_above_boundary & pixels_with_picture_neighbors) | (
            pixels_in_picture & pixels_with_boundary_neighbors
)


def my_check_interleave_pedestal_cleaning(path_list, calib_time_file, calib_file, noise_pixels_id_list, max_events=10000):

    signal_place_after_clean = np.zeros(1855)
    sum_ped_ev = 0
    alive_ped_ev = 0

    for path in path_list:
        print(path)
        r0_r1_calibrator = LSTR0Corrections(pedestal_path=None,
                                            r1_sample_start=3,
                                            r1_sample_end=39)

        r1_dl1_calibrator = LSTCameraCalibrator(calibration_path=calib_file,
                                                time_calibration_path=calib_time_file,
                                                extractor_product="LocalPeakWindowSum",
                                                config=charge_config,
                                                allowed_tels=[1])

        reader = LSTEventSource(input_url=path, max_events=max_events)


        for i, ev in enumerate(reader):
            r0_r1_calibrator.calibrate(ev)
            if i%10000 == 0:
                print(ev.r0.event_id)

            if ev.lst.tel[1].evt.tib_masked_trigger == 32:
                sum_ped_ev += 1

                r1_dl1_calibrator(ev)

                img = ev.dl1.tel[1].image

                geom = ev.inst.subarray.tel[1].camera
                clean = my_tailcuts_clean(
                                    geom,
                                    img,
                                    noise_pixels_id_list,
                                    picture_thresh=6,
                                    boundary_thresh=3,
                                    min_number_picture_neighbors=1,
                                    keep_isolated_pixels=False
                                    )

                cleaned = img.copy()
                cleaned[~clean] = 0.0

                signal_place_after_clean[np.where(clean == True)] += 1

                if np.sum(cleaned>0) > 0:
                    alive_ped_ev += 1

    fig, ax = plt.subplots(figsize=(8, 8))
    geom = ev.inst.subarray.tel[1].camera

    disp0 = CameraDisplay(geom, ax=ax)
    disp0.image = signal_place_after_clean/sum_ped_ev
    disp0.add_colorbar(ax=ax, label="N times signal remain after cleaning")
    disp0.cmap = 'gnuplot2'
    ax.set_title("{} \n {}/{}".format(path.split("/")[-1][8:21], alive_ped_ev, sum_ped_ev))

    print(path.split("/")[-1][8:21])
    print("{}/{}".format(alive_ped_ev, sum_ped_ev))

    ax.set_xlabel(" ")
    ax.set_ylabel(" ")
    plt.tight_layout()
    plt.show()

    return signal_place_after_clean, sum_ped_ev
