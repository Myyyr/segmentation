import json

from dataio.loader.ukbb_dataset import UKBBDataset
from dataio.loader.test_dataset import TestDataset
from dataio.loader.hms_dataset import HMSDataset
from dataio.loader.cmr_3D_dataset import CMR3DDataset
from dataio.loader.us_dataset import UltraSoundDataset
from dataio.loader.split_tcia_3D_dataset import SplitTCIA3DDataset


def get_dataset(name):
    """get_dataset

    :param name:
    """

    print("||| GET DATASET |||")
    return {
        'ukbb_sax': CMR3DDataset,
        'acdc_sax': CMR3DDataset,
        'rvsc_sax': CMR3DDataset,
        'hms_sax':  HMSDataset,
        'test_sax': TestDataset,
        'us': UltraSoundDataset,
        'split_tcia': SplitTCIA3DDataset
    }[name]


def get_dataset_path(dataset_name, opts):
    """get_data_path

    :param dataset_name:
    :param opts:
    """

    return getattr(opts, dataset_name)
