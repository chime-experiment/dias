class DiasException(Exception):
    """\
Dias base exception class
"""
    def __init__(self, message):
        self.message = message

