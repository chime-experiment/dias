"""Analyzer to retrieve and store daily ringmaps."""

import numpy as np
import requests
import msgpack
from os import path
import h5py

from dias import CHIMEAnalyzer
from dias.exception import DiasDataError
from caput import config
from ch_util import ephemeris as ephem
from bitshuffle import h5

TIME_DTYPE = np.dtype([("fpga_count", "<u8"), ("ctime", "<f8")])
POL_LABELS = ["XX", "XY", "YX", "YY"]


class DailyRingmapAnalyzer(CHIMEAnalyzer):
    """
    Analyzer to retrieve and store daily ringmaps.

    Metrics
    -------
    dias_task_<task_name>_maps_total
    .....................................
    The number of maps that were saved in the last run.

    Output Data
    -----------
    A file for every day containing ringmaps at all available
    frequencies and polarizations.

    File naming
    ...........
    `%Y%m%dT%H%M%SZ.h5`
        UTC time corresponding to first time sample in ringmap.

    Indexes
    .......
    time
        Time axis.
    pol
        Polarization.
    freq
        Frequency.
    sinza
        Sin of zenith angle.

    Datasets
    .........
    ringmap
        Ringmap.
    weight
        Summed inverse variance weights for every time sample.

    State Data
    ----------
    None

    Config
    ------
    ringmap_url : str
        URL of the ringmap server. (default http://recv1:12048/ringmap)
    use_bitshuffle : bool
        Whether to use bitshuffle to compress the data in the output HDF5 files
        (default False)

    Attributes
    ----------
    None
    """

    ringmap_url = config.Property(proptype=str, default="http://recv1:12048/ringmap")
    use_bitshuffle = config.Property(proptype=bool, default=False)

    def setup(self):
        """Set up metrics for task."""
        # TODO: Allow subset of frequencies?

        # metric for number of maps in last run
        self.num_maps_metric = self.add_data_metric(
            "maps", "Number of maps saved in last run.", unit="total"
        )
        super().setup()

    def run(self):
        """Retrieve and save ringmaps."""
        # reset maps metric
        self.num_maps_metric.set(0)

        # fetch available frequencies and polarizations
        data = self._check_request()
        if data is None:
            raise DiasDataError("Failed to retrieve ringmap axes from server.")
        try:
            freq = data["freq"]
            pol = np.array([POL_LABELS[p] for p in data["pol"]], dtype="S2")
        except KeyError:
            raise DiasDataError("Ringmap axes missing expected freq and pol values.")

        # Retrieve data from server and write to file
        fh = None
        try:
            for pi, p in enumerate(pol):
                for fi, f in enumerate(freq):
                    r = self._check_request(
                        data={"freq_ind": fi, "pol": pi}, return_raw=True
                    )

                    if r is None:
                        self.warn(
                            "Failed to fetch ringmap for pol {}, freq {}.".format(p, f)
                        )
                        continue

                    # Unpack data
                    try:
                        data = msgpack.unpackb(r, raw=False)
                    except Exception:
                        self.logger.warn(
                            "Failed to unpack data for pol {}, freq {}.".format(p, f)
                        )
                        continue

                    # Parse data
                    try:
                        sinza = np.array(data["sinza"], dtype=np.float32)
                        time = np.array(
                            [(t["fpga_count"], t["ctime"]) for t in data["time"]],
                            dtype=TIME_DTYPE,
                        )
                        rmap = np.array(data["ringmap"], dtype=np.float32)
                        rmap = rmap.reshape(rmap.shape[0] // len(sinza), len(sinza)).T
                        wgt = np.array(data["weight"], dtype=np.float32)
                    except KeyError as key_name:
                        self.logger.warn(
                            "Missing key in ringmap response: {}.".format(key_name)
                        )
                        continue

                    # reorder so that time is increasing
                    if fh is None:
                        # Use common axes for all maps
                        common_time = time.copy()
                        sort_t = np.argsort(time["ctime"])
                        # create file
                        fh = self._create_file(pol, freq, common_time[sort_t], sinza)

                    ## NOTE
                    ## ringmap maker doesn't guarantee a fixed-length array
                    ## until it's been fully filled in
                    ## if it was restarted recently; you may see failures
                    ## where arrays differ in size and arrays changed in size
                    ## this will self-resolve once the ringmap-maker is done
                    ##

                    # determine if some times were updated during requests
                    t_offset = time["ctime"] - common_time["ctime"]
                    if not (t_offset == 0.0).all():
                        # zero times that are different
                        new_t = t_offset != 0.0
                        rmap[:, new_t] = 0.0
                        wgt[new_t] = 0.0

                    # write to file
                    fh["ringmap"][pi, fi, :, :] = rmap[:, sort_t]
                    fh["weight"][pi, fi, :] = wgt[sort_t]
                    self.num_maps_metric.inc()
        finally:
            if fh is not None:
                fh.close()

        msg = "Saved daily ringmaps for {}.".format(self.tag)
        self.logger.debug(msg)
        return msg

    def _check_request(self, data={}, return_raw=False):
        url = self.ringmap_url
        try:
            if data:
                r = requests.post(url, json=data)
            else:
                r = requests.get(url)
        except requests.Timeout:
            self.logger.warn("Request for ringmap at {} timed out.".format(url))
            return None
        except requests.ConnectionError:
            self.logger.warn("Could not reach ringmap server at {}.".format(url))
            return None

        if not r.ok:
            self.logger.warn(
                "Bad response from {}: {} {}".format(url, r.status_code, r.reason)
            )
            return None

        if return_raw:
            return r.content
        else:
            return r.json()

    def _create_file(self, pol, freq, time, sinza):
        # Create new file
        self.tag = ephem.unix_to_datetime(time[0]["ctime"]).strftime("%Y%m%dT%H%M%SZ")
        fname = path.join(self.write_dir, "{}.h5".format(self.tag))
        fh = h5py.File(fname, "w")

        # compression
        comp = h5.H5FILTER if self.use_bitshuffle else None
        comp_opts = (0, h5.H5_COMPRESS_LZ4) if self.use_bitshuffle else None
        # Create datasets
        im = fh.create_group("index_map")
        im.create_dataset("time", data=time)
        im.create_dataset("pol", data=pol)
        im.create_dataset("freq", data=freq)
        im.create_dataset("sinza", data=sinza)
        fh.create_dataset(
            "ringmap",
            shape=(len(pol), len(freq), len(sinza), len(time)),
            dtype=np.float32,
            compression=comp,
            compression_opts=comp_opts,
        )
        fh.create_dataset(
            "weight",
            shape=(len(pol), len(freq), len(time)),
            dtype=np.float32,
            compression=comp,
            compression_opts=comp_opts,
        )
        fh["ringmap"].attrs["axes"] = ("pol", "freq", "sinza", "time")
        fh["weight"].attrs["axes"] = ("pol", "freq", "time")

        return fh
