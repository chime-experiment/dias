import datetime
import requests
import numpy as np

from caput import config
from dias import chime_analyzer
from dias.utils.string_converter import str2timedelta
import matplotlib as plt
from psd import PowerSpectralDensity

__version__ = "0.1.0"


class RxTempAnalyzer(chime_analyzer.CHIMEAnalyzer):

    period = config.Property(proptype=str2timedelta, default="24h")
    instrument = config.Property(proptype=str, default="chime")

    acqs_suffix = config.Property(proptype=str, default="hkp")

    def setup(self):
        """Set up the analyzer.

        Initialize Prometheus metrics.
        """
        self.logger.info(
            "Starting up. My name is %s and I am of type %s." % (self.name, __name__)
        )

    def load_temp(self):
        step = 30.0
        output = [dict(), dict()]
        for ind in range(2):
            output[ind]["time"] = []
            output[ind]["temp"] = []

        # Determine the range of time to process
        end_datetime = datetime.datetime.utcnow()
        start_datetime = end_datetime - self.period

        sts = start_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
        ets = end_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Query prometheus running on hk-east
        prom_query_east = 'ext_sensor_value{device="Ambient Temperature", type="temperature", job="environment_sensors", instance="e-enviromux"}/10'
        prom_query_west = 'ext_sensor_value{device="Ambient Temperature", type="temperature", job="environment_sensors", instance="w-enviromux"}/10'

        for ind, prom_query in enumerate([prom_query_east, prom_query_west]):

            http_query = (
                "http://hk-east:9090/api/v1/query_range?query=%s&start="
                "%s&end=%s&step=%dm" % (prom_query, sts, ets, step)
            )

            resp = requests.get(http_query)
            jresp = resp.json()

            for res in jresp["data"]["result"]:
                timestamp, temp = zip(*res["values"])
                output[ind]["time"] += timestamp
                output[ind]["temp"] += temp

            output[ind]["time"] = np.array(output[ind]["time"])
            output[ind]["temp"] = np.array(output[ind]["temp"])

        report = {}
        report["time"] = output[0]["time"]
        report["temp"] = np.vstack([output[0]["temp"], output[1]["temp"]])

        return report

    def run(self):
        """Run the analyzer.
        """
        report = self.load_temp()

        timestamp = report["time"]
        temp = report["temp"]

        temp_psd = PowerSpectralDensity(
            timestamp,
            temp,
            overlap_factor=0.5,
            tmax=0.1 * (timestamp[-1] - timestamp[0]),
            speedy=True,
            one_sided=True,
        )

    def finish(self):
        """Close connections."""
        self.logger.info("Shutting down.")
