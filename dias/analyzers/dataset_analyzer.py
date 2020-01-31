from dias import CHIMEAnalyzer

import numpy as np

from ch_util import andata

from chimedb import core
from chimedb import dataset as ds


def validate_flag_updates(ad, dsindex, flg, freq_id):

    extra_bad = [
      # These are non-CHIME feeds we want to exclude (26m, noise source channels, etc.)
        46,  142,  688,  944,  960, 1058, 1166, 1225, 1314, 1521, 2032, 2034,
      # Below are the last eight feeds on each cylinder, masked out because their beams are very different
         0,    1,    2,    3,    4,    5,    6,    7,  248,  249,  250,  251,  252,  253,  254,  255,
       256,  257,  258,  259,  260,  261,  262,  263,  504,  505,  506,  507,  508,  509,  510,  511,
       512,  513,  514,  515,  516,  517,  518,  519,  760,  761,  762,  763,  764,  765,  766,  767,
       768,  769,  770,  771,  772,  773,  774,  775, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023,
      1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279,
      1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535,
      1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791,
      1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047
    ]

    # Get the unique dataset_ids in each file
    file_ds = ad.flags["dataset_id"][:]
    unique_ds = np.unique(file_ds)

    # Find the flagging update_id for each dataset
    states = {
        id: (dsindex[bytes(id).decode()]
             .closest_ancestor_of_type("flags")
             .state.data["data"].encode())
        for id in unique_ds
    }

    ret = {}

    for ds_id, update_id in states.items():

        # Get all non-missing frames at this frequency
        present = (ad.flags["frac_lost"][freq_id] < 1.0)

        # Select all present frames with the dataset id we want
        sel = (file_ds[freq_id] == ds_id) & present

        # Extract the input flags and mark any extra missing data
        # (i.e. that static mask list in baselineCompression)
        flags = ad.flags["inputs"][:, sel].astype(np.bool)
        flags[extra_bad, :] = False

        # Find the flag update from the file
        flgind = list(flg.update_id).index(states[ds_id])
        flagsfile = flg.flag[flgind]
        flagsfile[extra_bad] = False

        # Test if all flag entries match the one from the flaginput file
        ret[ds_id] = (flags == flagsfile[:, np.newaxis]).all()

    return ret

class DatasetAnalyzer(CHIMEAnalyzer):

   def setup(self):
      self.logger.info(self.name + " setup")

   def run(self):
      self.logger.info(self.name + " running")

      # make chimedb connect
      core.connect()

      # pre-fetch most stuff to save queries later
      dsind = ds.get.index()


      fn1 = "/mnt/gong/archive/20191217T122901Z_chimestack_corr/00000000_0000.h5"
      fn2 = "/mnt/gong/archive/20191220T204152Z_chimestack_corr/00000000_0000.h5"

      ad1 = andata.CorrData.from_acq_h5(fn1, datasets=("flags/inputs", "flags/dataset_id", "flags/frac_lost"))
      ad2 = andata.CorrData.from_acq_h5(fn2, datasets=("flags/inputs", "flags/dataset_id", "flags/frac_lost"))

      flg = andata.FlagInputData.from_acq_h5("/mnt/gong/staging/20191201T000000Z_chime_flaginput/*.h5")

      self.logger.info(validate_flag_updates(ad1, dsind, flg, 10))

      self.logger.info(validate_flag_updates(ad2, dsind, flg, 10))
