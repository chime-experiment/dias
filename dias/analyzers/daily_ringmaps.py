import numpy as np
import requests
import msgpack
from datetime import datetime
from os import path
import h5py

from dias import CHIMEAnalyzer
from dias.exception import DiasDataError
from dias.utils import str2timedelta, datetime2str
from caput import config
from ch_util import ephemeris as ephem
from bitshuffle import h5

TIME_DTYPE = np.dtype([('fpga_count', '<u8'), ('ctime', '<f8')])


class DailyRingmapAnalyzer(CHIMEAnalyzer):

    ringmap_url = config.Property(proptype=str, default="http://recv1:12048/ringmap")
    use_bitshuffle = config.Property(proptype=bool, default=False)

    def setup(self):
        # TODO: Allow subset of frequencies?
        super().setup()

    def run(self):
        # fetch available frequencies and polarizations
        data = self._check_request()
        if data is None:
            raise DiasDataError("Failed to retrieve ringmap axes from server.")
        try:
            freq = data["freq"]
            pol = data["pol"]
        except KeyError:
            raise DiasDataError("Ringmap axes missing expected freq and pol values.")

        # Get all data from server before writing
        # This is to try and ensure the times are consistent between all freq/pol
        # If it turns out this strains the ringmap server too much, could merge loops
        # to increase time between requests, but will probably lead to mismatched time axes
        all_data = []
        for pi, p in enumerate(pol):
            for fi, f in enumerate(freq):
                r = self._check_request(data={"freq_ind": fi, "pol": pi}, return_raw=True)
                all_data.append(r)

        # Check that some data was successfully retrieved
        got_something = False
        for d in data:
            if d is not None:
                got_something = True
                break
        if not got_something:
            raise DiasDataError("Failed to retrieve any ringmaps from server.")

        # write to file
        fh = None
        try:
            for pi, p in enumerate(pol):
                for fi, f in enumerate(freq):
                    i = pi * len(freq) + fi

                    if all_data[i] is None:
                        self.warn("Failed to fetch ringmap for pol {}, freq {}.".format(p, f))
                        # TODO: record in prometheus

                    # Unpack data
                    try:
                        data = msgpack.unpackb(all_data[i], raw=False)
                    except:
                        self.logger.warn("Failed to unpack data for pol {}, freq {}.".format(p, f))
                        # TODO: record in prometheus
                        continue

                    # Parse data
                    try:
                        sinza = np.array(data["sinza"], dtype=np.float32)
                        time = np.array([(t["fpga_count"], t["ctime"]) for t in data["time"]],
                                        dtype=TIME_DTYPE)
                        rmap = np.array(data["ringmap"], dtype=np.float32)
                        rmap = rmap.reshape(rmap.shape[0] // len(sinza), len(sinza)).T
                        wgt = np.array(data["weight"], dtype=np.float32)
                    except KeyError:
                        self.logger.warn("Missing keys in ringmap response.")
                        # TODO: record in prometheus
                        continue

                    # reorder so that time is increasing
                    sort_t = np.argsort(time["ctime"])
                    if fh is None:
                        # Use common axes for all maps
                        common_time = time
                        # create file
                        fh = self._create_file(pol, freq, common_time[sort_t], sinza)

                    # determine if some times were updated during requests
                    t_offset = time["ctime"] - common_time["ctime"]
                    if not (t_offset == 0.).all():
                        # zero times that are different
                        new_t = t_offset != 0.
                        rmap[:, new_t[sort_t]] = 0.
                        wgt[new_t[sort_t]] = 0.

                    # write to file
                    fh["ringmap"][pi, fi, :, :] = rmap[:, sort_t]
                    fh["weight"][pi, fi, :] = wgt[sort_t]
        finally:
            if fh is not None:
                fh.close()

        msg = "Saved daily ringmaps for {}.".format(self.tag)
        self.logger.debug(msg)
        # TODO: return more informative summary
        return msg


    def delete_callback(self, deleted_files):
        pass

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
            self.logger.warn("Bad response from {}: {} {}".format(url, r.status_code, r.reason))
            return None

        if return_raw:
            return r.content
        else:
            return r.json()

    def _create_file(self, pol, freq, time, sinza):
        # Create new file
        self.tag = ephem.unix_to_datetime(time[0]['ctime']).strftime("%Y%m%dT%H%M%SZ")
        fname = path.join(self.write_dir, "{}.h5".format(self.tag))
        fh = h5py.File(fname)

        # compression
        comp = h5.H5FILTER if self.use_bitshuffle else None
        comp_opts = (0, h5.H5_COMPRESS_LZ4) if self.use_bitshuffle else None
        # Create datasets
        im = fh.create_group("index_map")
        im.create_dataset("time", data=time)
        im.create_dataset("pol", data=pol)
        im.create_dataset("freq", data=freq)
        im.create_dataset("sinza", data=sinza)
        fh.create_dataset("ringmap", shape=(len(pol), len(freq), len(sinza), len(time)),
                          dtype=np.float32, compression=comp, compression_opts=comp_opts)
        fh.create_dataset("weight", shape=(len(pol), len(freq), len(time)),
                          dtype=np.float32, compression=comp, compression_opts=comp_opts)
        fh["ringmap"].attrs["axes"] = ("pol", "freq", "sinza", "time")
        fh["weight"].attrs["axes"] = ("pol", "freq", "time")

        return fh
