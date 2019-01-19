# CHIME-specifc Analyzer Base Class
# --------------------------------

from dias.analyzer import Analyzer
from ch_util import data_index
from caput import config

class CHIMEAnalyzer(Analyzer):
    """A base dias analyzer class with CHIME-specific extensions.

    The CHIMEAnalyzer class extends the base dias Analyzer class by adding
    CHIME-specific convenience methods.
    """

    archive_data_dir = config.Property(proptype=str)

    def Finder(self, acqs=()):
        """Returns a ch_util Finder object for use by Analyzer tasks
        """

        return data_index.Finder(acqs=acqs,
                node_spoof = { "gong" : self.archive_data_dir })
