# CHIME-specifc Analyzer Base Class
# --------------------------------

from dias.analyzer import Analyzer
from ch_util import data_index
from caput import config
from dias.config_loader import DEFAULT_ARCHIVE_DIR

class CHIMEAnalyzer(Analyzer):
    """A base dias analyzer class with CHIME-specific extensions.

    The CHIMEAnalyzer class extends the base dias Analyzer class by adding
    CHIME-specific convenience methods.
    """

    archive_data_dir = config.Property(proptype=str,
                                       default=DEFAULT_ARCHIVE_DIR)

    def Finder(self, acqs=()):
        """Returns a ch_util Finder object for use by Analyzer tasks
        """

        return data_index.Finder(acqs=acqs,
                node_spoof = { "gong" : self.archive_data_dir })
