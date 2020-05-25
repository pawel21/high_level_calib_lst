import h5py
import tables
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numba import jit, njit, prange

from ctapipe.core import Component
from traitlets.config import Config
from ctapipe.core.traits import Int, Float, Unicode
from ctapipe.image.extractor import ImageExtractor
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from ctapipe.image.cleaning import tailcuts_clean

from ctapipe_io_lst import LSTEventSource
from lstchain.calib.camera.r0 import LSTR0Corrections
from lstchain.calib.camera.calibrator import LSTCameraCalibrator

__all__ = ['CleanigPedestalImage']

plt.rcParams.update({'font.size': 22})

high_gain = 0
low_gain = 1

n_gain = 2
n_channel = 7
n_modules = 265
n_pixels = 1855


class CleanigPedestalImage(Component):
    """
        Class to chceck pedestal image
    """

    tel_id = Int(1,
                 help='Id of the telescope to calibrate'
                 ).tag(config=True)



    charge_product = Unicode(
        'LocalPeakWindowSum',
        help='Name of the charge extractor to be used'
    ).tag(config=True)


    calib_file = Unicode('',
                        allow_none=True,
                        help='Path to the calibration file'
                        ).tag(config=True)

    calib_time_file = Unicode('',
                              allow_none=True,
                              help="Path to the time calibration file"
                              ).tag(config=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cleaning_parameters = self.config["tailcut"]
        print(self.config)
        self.r1_dl1_calibrator = LSTCameraCalibrator(
                                                calibration_path=self.calib_file,
                                                time_calibration_path=self.calib_time_file,
                                                extractor_product=self.charge_product,
                                                config=self.config,
                                                gain_threshold = Config(self.config).gain_selector_config['threshold'],
                                                allowed_tels=[1]
                                                )


    def run(self, list_of_file, max_events):

        signal_place_after_clean = np.zeros(1855)
        sum_ped_ev = 0
        alive_ped_ev = 0


        for input_file in list_of_file:
            print(input_file)

            r0_r1_calibrator = LSTR0Corrections(pedestal_path=None,
                                                r1_sample_start=3,
                                                r1_sample_end=39)
            reader = LSTEventSource(input_url=input_file,
                                    max_events=max_events)
            for i, ev in enumerate(reader):
                r0_r1_calibrator.calibrate(ev)
                if i%10000 == 0:
                    print(ev.r0.event_id)

                if ev.lst.tel[1].evt.tib_masked_trigger == 32:
                    sum_ped_ev += 1
                    self.r1_dl1_calibrator(ev)

                    img = ev.dl1.tel[1].image

                    geom = ev.inst.subarray.tel[1].camera
                    clean = tailcuts_clean(
                                        geom,
                                        img,
                                        **self.cleaning_parameters
                                        )

                    cleaned = img.copy()
                    cleaned[~clean] = 0.0

                    signal_place_after_clean[np.where(clean == True)] += 1

                    if np.sum(cleaned>0) > 0:
                        alive_ped_ev += 1

        fig, ax = plt.subplots(figsize=(10, 8))
        geom = ev.inst.subarray.tel[1].camera

        disp0 = CameraDisplay(geom, ax=ax)
        disp0.image = signal_place_after_clean/sum_ped_ev
        disp0.add_colorbar(ax=ax, label="N times signal remain after cleaning [%]")
        disp0.cmap = 'gnuplot2'
        ax.set_title("{} \n {}/{}".format(input_file.split("/")[-1][8:21], alive_ped_ev, sum_ped_ev), fontsize=25)

        print("{}/{}".format(alive_ped_ev, sum_ped_ev))

        ax.set_xlabel(" ")
        ax.set_ylabel(" ")
        plt.tight_layout()
        plt.show()


    def remove_star_and_run(self, list_of_file, max_events, noise_pixels_id_list):
        signal_place_after_clean = np.zeros(1855)
        sum_ped_ev = 0
        alive_ped_ev = 0

        for input_file in list_of_file:
            print(input_file)

            r0_r1_calibrator = LSTR0Corrections(pedestal_path=None,
                                                r1_sample_start=3,
                                                r1_sample_end=39)
            reader = LSTEventSource(input_url=input_file,
                                    max_events=max_events)
            for i, ev in enumerate(reader):
                r0_r1_calibrator.calibrate(ev)
                if i%10000 == 0:
                    print(ev.r0.event_id)

                if ev.lst.tel[1].evt.tib_masked_trigger == 32:
                    sum_ped_ev += 1
                    self.r1_dl1_calibrator(ev)

                    img = ev.dl1.tel[1].image
                    img[noise_pixels_id_list] = 0

                    geom = ev.inst.subarray.tel[1].camera
                    clean = tailcuts_clean(
                                        geom,
                                        img,
                                        **self.cleaning_parameters
                                        )

                    cleaned = img.copy()
                    cleaned[~clean] = 0.0

                    signal_place_after_clean[np.where(clean == True)] += 1

                    if np.sum(cleaned>0) > 0:
                        alive_ped_ev += 1

        fig, ax = plt.subplots(figsize=(10, 8))
        geom = ev.inst.subarray.tel[1].camera

        disp0 = CameraDisplay(geom, ax=ax)
        disp0.image = signal_place_after_clean/sum_ped_ev
        disp0.highlight_pixels(noise_pixels_id_list, linewidth=3)
        disp0.add_colorbar(ax=ax, label="N times signal remain after cleaning [%]")
        disp0.cmap = 'gnuplot2'
        ax.set_title("{} \n {}/{}".format(input_file.split("/")[-1][8:21], alive_ped_ev, sum_ped_ev), fontsize=25)

        print("{}/{}".format(alive_ped_ev, sum_ped_ev))

        ax.set_xlabel(" ")
        ax.set_ylabel(" ")
        plt.tight_layout()
        plt.show()

    def plot_camera_display(self, image, input_file, noise_pixels_id_list, alive_ped_ev, sum_ped_ev):
        fig, ax = plt.subplots(figsize=(10, 8))
        geom = CameraGeometry.from_name('LSTCam-003')

        disp0 = CameraDisplay(geom, ax=ax)
        disp0.image = image
        disp0.highlight_pixels(noise_pixels_id_list, linewidth=3)
        disp0.add_colorbar(ax=ax, label="N times signal remain after cleaning [%]")
        disp0.cmap = 'gnuplot2'
        ax.set_title("{} \n {}/{}".format(input_file.split("/")[-1][8:21], alive_ped_ev, sum_ped_ev), fontsize=25)

        print("{}/{}".format(alive_ped_ev, sum_ped_ev))

        ax.set_xlabel(" ")
        ax.set_ylabel(" ")
        plt.tight_layout()
        plt.show()

    def check_interleave_pedestal_cleaning(self,
                                           list_of_file,
                                           max_events,
                                           sigma,
                                           dl1_file):

        high_gain = 0
        ped_mean_pe, ped_rms_pe = get_bias_and_rms(dl1_file)
        bad_pixel_ids = np.where(ped_rms_pe[1, high_gain, :] == 0)[0]
        print(bad_pixel_ids)
        th = get_threshold(ped_mean_pe[1, high_gain, :],
                           ped_rms_pe[1, high_gain, :],
                           sigma)

        make_camera_binary_image(th,
                                 sigma,
                                 self.cleaning_parameters['picture_thresh'],
                                 bad_pixel_ids)

        signal_place_after_clean = np.zeros(1855)
        sum_ped_ev = 0
        alive_ped_ev = 0

        for input_file in list_of_file:
            print(input_file)

            r0_r1_calibrator = LSTR0Corrections(pedestal_path=None,
                                                r1_sample_start=3,
                                                r1_sample_end=39)

            reader = LSTEventSource(input_url=input_file,
                                    max_events=max_events)

            for i, ev in enumerate(reader):
                r0_r1_calibrator.calibrate(ev)
                if i%10000 == 0:
                    print(ev.r0.event_id)

                if ev.lst.tel[1].evt.tib_masked_trigger == 32:
                    sum_ped_ev += 1
                    self.r1_dl1_calibrator(ev)

                    img = ev.dl1.tel[1].image
                    img[bad_pixel_ids] = 0
                    geom = ev.inst.subarray.tel[1].camera
                    clean = tailcuts_pedestal_clean(
                                        geom,
                                        img,
                                        th,
                                        **self.cleaning_parameters
                                        )

                    cleaned = img.copy()
                    cleaned[~clean] = 0.0

                    signal_place_after_clean[np.where(clean == True)] += 1
                    if np.sum(cleaned>0) > 0:
                        alive_ped_ev += 1

        noise_remain = signal_place_after_clean/sum_ped_ev

        self.plot_camera_display(noise_remain,
                                 input_file,
                                 bad_pixel_ids,
                                 alive_ped_ev,
                                 sum_ped_ev)



def tailcuts_pedestal_clean(
    geom,
    image,
    ped_threshold,
    picture_thresh=7,
    boundary_thresh=5,
    keep_isolated_pixels=False,
    min_number_picture_neighbors=0,
):

    pixels_above_picture = np.logical_and(image>= picture_thresh, image >= ped_threshold)

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

def get_bias_and_rms(dl1_file):
    "return bias (mean) and rms from pedestal events in pe"

    f = tables.open_file(dl1_file)
    ped = f.root['/dl1/event/telescope/monitoring/pedestal']
    ped_charge_mean = np.array(ped.cols.charge_mean)
    ped_charge_rms = np.array(ped.cols.charge_std)
    calib = f.root['/dl1/event/telescope/monitoring/calibration']
    dc_to_pe = np.array(calib.cols.dc_to_pe)
    ped_charge_mean_pe = ped_charge_mean*dc_to_pe
    ped_charge_rms_pe = ped_charge_rms*dc_to_pe
    f.close()

    return ped_charge_mean_pe, ped_charge_rms_pe

def make_camera_image(image):
    plt.figure(figsize=(6, 6))
    geom = CameraGeometry.from_name('LSTCam-003')
    disp = CameraDisplay(geom)
    disp.image = image
    disp.cmap = plt.cm.gnuplot2
    disp.add_colorbar()
    plt.show()

def get_threshold(ped_mean_pe, ped_rms_pe, sigma_clean):
    threshold_clean_pe = ped_mean_pe + sigma_clean*ped_rms_pe
    return threshold_clean_pe

def make_camera_binary_image(image, sigma, clean_bound, death_pixel_ids_list):
    fig, ax = plt.subplots(1,2, figsize=(14, 7))

    geom = CameraGeometry.from_name('LSTCam-003')
    disp0 = CameraDisplay(geom, ax=ax[0])
    disp0.image = image
    disp0.cmap = plt.cm.gnuplot2
    disp0.add_colorbar(ax=ax[0])
    ax[0].set_title("Cleaning threshold from interleaved pedestal. \n sigma = {}".format(sigma), fontsize=15)

    disp1 = CameraDisplay(geom, ax=ax[1])
    disp1.image = image
    cmap = matplotlib.colors.ListedColormap(['black', 'red'])
    bounds=[0, clean_bound, 2*clean_bound]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    disp1.cmap = cmap
    disp1.add_colorbar(norm=norm, boundaries=bounds, ticks=[0, clean_bound, 2*clean_bound])
    disp1.set_limits_minmax(0, 2*clean_bound)
    disp1.highlight_pixels(death_pixel_ids_list, linewidth=3)
    ax[1].set_title("Red pixels - above cleaning tailcut threshold", fontsize=15)
    plt.tight_layout()
    plt.show()


def check_interleave_pedestal_cleaning():
    ped_mean_pe, ped_rms_pe = get_bias_and_rms("../data/data_dl1/20191124/dl1_data/dl1_LST-1.Run01627.0002.h5")
    th = get_threshold(ped_mean_pe[1, 0, :], ped_rms_pe[1, 0, :], 3)
    print(th)
    make_camera_image(th)
