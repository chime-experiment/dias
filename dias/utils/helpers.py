"""Misc helper functions that are used by multiple analyzers."""


def get_cyl(cyl_num, cyl_start_num, cyl_start_char):
    """Return the cylinder ID (char)."""
    return chr(cyl_num - cyl_start_num + cyl_start_char)
