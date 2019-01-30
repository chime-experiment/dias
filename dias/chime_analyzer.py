# CHIME-specifc Analyzer Base Class
# --------------------------------

from dias.analyzer import Analyzer
from ch_util import data_index

DEFAULT_ARCHIVE_DIR = ''


class CHIMEAnalyzer(Analyzer):
    """A base dias analyzer class with CHIME-specific extensions.

    The CHIMEAnalyzer class extends the base dias Analyzer class by adding
    CHIME-specific convenience methods.
    """

    def __init__(self, name, config, write_dir, state_dir):
        """Constructor of CHIME analyzer base class.
        """
        self.archive_data_dir = config.read_config_variable(
            'archive_data_dir', proptype=str, default=DEFAULT_ARCHIVE_DIR)
        Analyzer.__init__(self, name, config, write_dir, state_dir)

    def Finder(self, acqs=()):
        """Returns a ch_util Finder object for use by Analyzer tasks
        """

        return data_index.Finder(
            acqs=acqs, node_spoof={"gong": self.archive_data_dir})
