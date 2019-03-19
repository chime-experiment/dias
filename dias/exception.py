"""dias exceptions."""
class DiasException(Exception):
    """Dias base exception class."""

    pass


class DiasUsageError(DiasException):
    """
    Exception raised for errors in the usage of dias.

    This is thrown when the user did something wrong typing the dias
    command-line options.

    Parameters
    ----------
    message : String
        Explanation of the error.
    """

    def __init__(self, message):
        self.message = message


class DiasConfigError(DiasException):
    """
    Exception raised for errors in the dias config.

    This is thrown when there is an error in the YAML config or task files.

    Parameters
    ----------
    message : String
        Explanation of the error.
    """

    def __init__(self, message):
        self.message = message
