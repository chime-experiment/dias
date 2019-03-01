class DiasException(Exception):
    """\
Dias base exception class
"""
    pass

class DiasUsageError(DiasException):
    """\
Exception raised for errors in the usage of dias.
:param message: Explanation of the error.
"""
    def __init__(self, message):
        self.message = message


class DiasConfigError(DiasException):
    """\
Exception raised for errors in the dias config..
:param message: Explanation of the error.
"""
    def __init__(self, message):
        self.message = message
