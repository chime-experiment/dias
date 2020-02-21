def get_cyl(cyl_num, cyl_start_num, cyl_start_char):
    """Return the cylinder ID (char)."""
    return chr(cyl_num - cyl_start_num + cyl_start_char)
