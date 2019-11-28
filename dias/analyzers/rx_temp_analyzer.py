import datetime
import requests

from caput import config
from dias import chime_analyzer
from dias.utils.string_converter import str2timedelta

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

    def run(self):
        """Run the analyzer.
        """
        step = 30.0

        # Determine the range of time to process
        end_datetime = datetime.datetime.utcnow()
        start_datetime = end_datetime - self.period

        sts = start_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
        ets = end_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Query prometheus running on hk-east
        prom_query = 'ext_sensor_value{device="Ambient Temperature", type="temperature", job="environment_sensors", instance="e-enviromux"}/10'

        http_query = (
            "http://hk-east:9090/api/v1/query_range?query=%s&start="
            "%s&end=%s&step=%dm" % (prom_query, sts, ets, step)
        )

        resp = requests.get(http_query)
        jresp = resp.json()

        for res in jresp["data"]["result"]:
            print(res)

    def finish(self):
        """Close connections."""
        self.logger.info("Shutting down.")
