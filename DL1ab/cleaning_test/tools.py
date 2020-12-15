import h5py
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tables
import astropy.units as u

from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from ctapipe.image.cleaning import tailcuts_clean
from traitlets.config.loader import Config
from ctapipe.core import Container, Field
from ctapipe.io.hdf5tableio import HDF5TableWriter, HDF5TableReader
from ctapipe.image.morphology import number_of_islands
from ctapipe.image import (
        hillas_parameters,
        tailcuts_clean,
        apply_time_delta_cleaning,
        HillasParameterizationError,
        )
from lstchain.io.config import read_configuration_file
from lstchain.io.io import read_camera_geometries, read_single_camera_geometry
from lstchain.io import DL1ParametersContainer

from cleaning import tailcuts_clean_with_pedestal_threshold, get_threshold_from_dl1_file


dl1_params_lstcam_key = 'dl1/event/telescope/parameters/LST_LSTCam'
geom = CameraGeometry.from_name('LSTCam-002')
foclen = 28*u.m

class CleaningInfo(Container):
    name = Field(str, "Name of cleaning method")
    pic_th = Field(float, 'Picture threshold value')
    bound_th = Field(float, 'Boundary threshold value')


def get_dl1b_tailcut(dl1a_img, dl1a_pulse, config_path, use_main_island=True):
    cleaning_method = tailcuts_clean
    config = read_configuration_file(config_path)
    cleaning_parameters = config["tailcut"]

    dl1_container = DL1ParametersContainer()

    image = dl1a_img
    pulse_time = dl1a_pulse
    signal_pixels = cleaning_method(camera_geometry, image, **cleaning_parameters)

    n_pixels = np.count_nonzero(signal_pixels)

    if n_pixels > 0:
        # check the number of islands
        num_islands, island_labels = number_of_islands(camera_geometry, signal_pixels)

        if use_main_island:
            n_pixels_on_island = np.bincount(island_labels.astype(np.int))
            n_pixels_on_island[0] = 0  # first island is no-island and should not be considered
            max_island_label = np.argmax(n_pixels_on_island)
            signal_pixels[island_labels != max_island_label] = False

        hillas = hillas_parameters(camera_geometry[signal_pixels], image[signal_pixels])

        dl1_container.fill_hillas(hillas)
        dl1_container.set_timing_features(camera_geometry[signal_pixels],
                                          image[signal_pixels],
                                          pulse_time[signal_pixels],
                                          hillas)

        set_converted_hillas_param(dl1_container, dl1_container.width, dl1_container.length)
        set_image_param(dl1_container, image, signal_pixels, hillas, n_pixels, num_islands)

    return dl1_container

def set_converted_hillas_param(dl1_container, width, length):
    # convert ctapipe's width and length (in m) to deg:
    width = np.rad2deg(np.arctan2(dl1_container.width, foclen))
    length = np.rad2deg(np.arctan2(dl1_container.length, foclen))
    dl1_container.width = width
    dl1_container.length = length
    dl1_container.wl = dl1_container.width / dl1_container.length

def set_image_param(dl1_container, image, signal_pixels, hillas, n_pixels, num_islands):
    dl1_container.set_leakage(camera_geometry, image, signal_pixels)
    dl1_container.set_concentration(camera_geometry, image, hillas)
    dl1_container.n_pixels = n_pixels
    dl1_container.n_islands = num_islands
    dl1_container.log_intensity = np.log10(dl1_container.intensity)

def get_trigger_info(dl1_file):
    f = tables.open_file(dl1_file)
    param = f.root['/dl1/event/telescope/parameters/LST_LSTCam']
    trigger_type = param.col("trigger_type")
    ucts_trigger_type = param.col("ucts_trigger_type")
    trigger_time = param.col("trigger_time")
    f.close()
    return trigger_type, ucts_trigger_type, trigger_time

def set_trigger_info(dl1b_container, trig_type, ucts_trig_type, trig_time):
    dl1b_container.trigger_type = trig_type
    dl1b_container.ucts_trigger_type = ucts_trig_type
    dl1b_container.trigger_time = trig_time

def get_time_info(dl1_file):
    f = tables.open_file(dl1_file)
    param = f.root['/dl1/event/telescope/parameters/LST_LSTCam']
    dragon_t = param.col("dragon_time")
    ucts_t = param.col("ucts_time")
    tib_t = param.col("tib_time")
    f.close()
    return dragon_t, ucts_t, tib_t

def set_time_info(dlb_container, dragon_t, ucts_t, tib_t):
    dlb_container.dragon_time = dragon_t
    dlb_container.ucts_time = ucts_t
    dlb_container.tib_time = tib_t

def get_id_info(dl1_file):
    f = tables.open_file(dl1_file)
    param = f.root['/dl1/event/telescope/parameters/LST_LSTCam']
    obs_id_array = param.col('obs_id')
    event_id_array = param.col("event_id")
    f.close()
    return obs_id_array, event_id_array

def set_id_info(dl1b_container, obs_id, event_id):
    dl1b_container.obs_id = obs_id
    dl1b_container.event_id = event_id


def create_dl1b_tailcut(dl1_path, output_file, config_file):
    global camera_geometry
    camera_geometry = read_single_camera_geometry(dl1_path, "LSTCam")

    obs_id_array, event_id_array = get_id_info(dl1_path)
    trig_type_array, ucts_trig_type_array, trig_time_array = get_trigger_info(dl1_path)
    dragon_t, ucts_t, tib_t = get_time_info(dl1_path)

    f = tables.open_file(dl1_path)
    dl1a_images = f.root['/dl1/event/telescope/image/LST_LSTCam'].col('image')
    dl1a_pulse = f.root['/dl1/event/telescope/image/LST_LSTCam'].col('peak_time')
    f.close()
    tel_name = "LST_LSTCam"
    clean_info = CleaningInfo()
    conf_f = read_configuration_file("lstchain_standard_config.json")
    print(conf_f)
    clean_info.name = "tailcut"
    clean_info.pic_th = conf_f["tailcut"]["picture_thresh"]
    clean_info.bound_th = conf_f["tailcut"]["boundary_thresh"]

    with HDF5TableWriter(
            filename=output_file,
            group_name='dl1/event',
            mode='a',
            filters=None,
            add_prefix=False
            ) as writer:
                for i in range(len(dl1a_images)):
                    dl1b = get_dl1b_tailcut(dl1a_images[i], dl1a_pulse[i], config_file)

                    set_trigger_info(dl1b, trig_type_array[i], ucts_trig_type_array[i], trig_time_array[i][0])
                    set_time_info(dl1b, dragon_t[i], ucts_t[i], tib_t[i])
                    set_id_info(dl1b, obs_id_array[i], event_id_array[i])
                    writer.write(table_name=f'telescope/parameters/{tel_name}',
                                 containers=dl1b)

                writer.write(table_name=f'telescope/parameters/info',
                             containers=clean_info)


def get_pedestal_images_and_pulse(dl1_filename):
    #trigger_source='trigger_type'
    trigger_source='ucts_trigger_type'
    f = tables.open_file(dl1_filename)
    parameters = pd.read_hdf(dl1_filename, key=dl1_params_lstcam_key)
    image_table = f.root.dl1.event.telescope.image.LST_LSTCam
    params_pedestal_mask = (parameters[trigger_source] == 32)
    ped_indices = np.array(parameters['event_id'][params_pedestal_mask])
    pedestal_mask = (parameters[trigger_source] == 32)
    ped_image =  image_table.col('image')[pedestal_mask]
    ped_pulse = image_table.col("peak_time")[pedestal_mask]
    f.close()
    return ped_image, ped_pulse
