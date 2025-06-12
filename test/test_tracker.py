"""Tests for dias.utils.Tracker.

Run from dias root with `pytest test/test_tracker.py
"""

import os
import h5py
import pytest
import numpy as np
from datetime import datetime
from pathlib import Path
import shutil
import yaml

from dias.utils import Tracker

base_path = os.path.expanduser("~/dias_tmp")
Path(base_path).mkdir(exist_ok=True)


@pytest.fixture(scope="session")
def staging_dir():
    """Create staging directory for tests."""
    staging = Path(os.path.join(base_path, "staging"))
    staging.mkdir(exist_ok=True)
    yield staging
    shutil.rmtree(staging)


@pytest.fixture()
def chimecal_testfolder(staging_dir):
    """Create chimecal testdata folder."""
    folder = os.path.join(staging_dir, "2020Z_chimecal_corr")
    return folder


@pytest.fixture()
def chimestack_testfolder(staging_dir):
    """Create chimestack testdata folder."""
    folder = os.path.join(staging_dir, "2020Z_chimestack_corr")
    return folder


@pytest.fixture
def testdata(chimecal_testfolder, chimestack_testfolder):
    """Create all of the testdata."""
    for folder in [chimecal_testfolder, chimestack_testfolder]:
        Path(folder).mkdir(exist_ok=True)
        for i in range(20):
            with h5py.File("{0}/sample_{1}.h5".format(folder, i), "w") as hf:
                index_map = hf.create_group("index_map")

                arr = np.rec.fromarrays(
                    [np.arange(1596549345.2508698, 1596551880.0784922)],
                    dtype=np.dtype({"names": ["ctime"], "formats": [float]}),
                )

                index_map.create_dataset("time", data=arr)

        # Lock some of the files
        for i in range(10):
            open(f"{folder}/.sample_{i+10}.h5.lock", "w").close()


@pytest.fixture(scope="function")
def reset_file_index():
    """Erase the file tracking index, so each test has a fresh one."""
    try:
        Path("{0}/file_index.db".format(base_path)).unlink()
    except FileNotFoundError:
        pass


@pytest.fixture
def file_index():
    """Return the location of the file tracking index from the config."""
    with open("./conf/dias.conf", "r") as f:
        dias_conf = yaml.safe_load(f)
    trackers = dias_conf["trackers"]
    for t in trackers:
        if t["name"] == "staging":
            db_file = t["db_file"]
    return db_file


def test_new_files_and_register_done(reset_file_index, testdata, file_index):
    """Test new_files and register_done for a single filetype."""
    client = Tracker("{0}/staging".format(base_path), file_index, write=True)

    my_todo = client.new_files("test_analyzer_1", filetypes="chimecal_corr")
    assert len(my_todo) == 10

    client.register_done("test_analyzer_1", my_todo)

    my_todo = client.new_files("test_analyzer_1", filetypes="chimecal_corr")
    assert len(my_todo) == 0


def test_multiple_filetypes(reset_file_index, testdata, file_index):
    """Test new_files and a partial register_done for multiple filetypes."""
    client = Tracker("{0}/staging".format(base_path), file_index, write=True)
    my_todo = client.new_files(
        "test_analyzer_2", filetypes=["chimecal_corr", "chimestack_corr"]
    )
    assert len(my_todo) == 20

    done = [f for f in my_todo if "chimecal" in f]

    client.register_done("test_analyzer_2", done)

    my_todo = client.new_files(
        "test_analyzer_2", filetypes=["chimecal_corr", "chimestack_corr"]
    )
    assert len(my_todo) == 10


def test_time_filter(reset_file_index, testdata, file_index):
    """Test new_files return of files intersecting within a timerange."""
    client = Tracker("{0}/staging".format(base_path), file_index, write=True)

    my_todo = client.new_files(
        "test_analyzer_3", filetypes="chimecal_corr", start=1593548300, end=1597551900
    )

    assert len(my_todo) == 10

    my_todo = client.new_files(
        "test_analyzer_3", filetypes="chimecal_corr", start=1597551900
    )

    assert len(my_todo) == 0


def test_output_files(reset_file_index, file_index):
    """Test the tracking of output files."""
    client = Tracker("{0}/staging".format(base_path), file_index, write=True)

    for i in range(10):
        filepath = "{0}/staging/{1}.txt".format(base_path, i)
        with open(filepath, "w") as f:
            f.write("test")
        client.add_output_file(
            "test_analyzer_4", 1596549345.2508698, 1596551880.0784922, filepath
        )

    todo = client.get_output_files("test_analyzer_4", start=1593548300, end=1597551900)

    assert len(todo) == 10

    todo = client.get_output_files("test_analyzer_4", start=1597551900)

    assert len(todo) == 0
