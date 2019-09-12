"""CHIME-specifc Analyzer Base Class."""

from caput import config
from dias import Analyzer
from ch_util import finder


class CHIMEAnalyzer(Analyzer):
    """
    A base dias analyzer class with CHIME-specific extensions.

    The CHIMEAnalyzer class extends the base dias Analyzer class by adding
    CHIME-specific convenience methods.
    """

    archive_data_dir = config.Property(proptype=str)

    def __init__(self, name, write_dir, state_dir):
        """Construct the CHIME analyzer base class.

        Parameters
        ----------
        name : String
            The name of the task.
        write_dir : String
            The path to write output data to.
        state_dir : String
            The path to write state data to.
        """
        super().__init__(name, write_dir, state_dir)

    def Finder(self, acqs=()):
        """
        Get a ch_util Finder object.

        Parameters
        ----------
        acqs : list of :class:`ArchiveAcq` objects
            Acquisition filter. Default: `()`.

        Returns
        -------
        An instance of :class:`Finder`.
        """
        return finder.Finder(
            acqs=acqs, node_spoof={"gong": self.archive_data_dir}
        )
