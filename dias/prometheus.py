"""
HTTP Server and clients for allowing Prometheus to access the metrics generated
by dias.
"""

import socket
import logging

from prometheus_client import start_http_server
from prometheus_client.core import GaugeMetricFamily


class Prometheus():
    """Prometheus client wrapper for dias.
    """

    def __init__(self, port):
        # Get the host name of this machine
        self.host = socket.gethostname()

        # Set a server to export (expose to prometheus) the data (in a thread)
        start_http_server(port)

        # Set the module logger.
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.info('Starting prometheus client on port {}.'
                         .format(str(port)))

    def add_metric(self, metric_name, value, documentation='',
                    timestamp=None, labels=dict(), unit=''):
        """Add a prometheus metric. Private method for class internal use.
        Instead call task_metric or data_metric methods of the analyzer base
         class."""
        # TODO: find a way to add a past timestamp for data metrics

        labels['host'] = self.host

        c = GaugeMetricFamily(metric_name, documentation, value=None,
                              labels=None, unit=unit)
        c.add_metric(labels, value, timestamp)
