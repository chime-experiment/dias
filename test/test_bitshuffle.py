import h5py
import numpy as np
from bitshuffle import h5

def test_bitshuffle():
        print("Testing bitshuffle import.")
        print("h5.H5_COMPRESS_LZ4: {}, h5.H5FILTER: {}".format(h5.H5_COMPRESS_LZ4, h5.H5FILTER))
  
        # Create new file
        fh = h5py.File("test.h5", "w")

        # compression
        comp = h5.H5FILTER
        comp_opts = (0, h5.H5_COMPRESS_LZ4)
        
        time = np.ones(1024)
        pol = np.ones(1024)
        freq = np.ones(1024)
        sinza = np.ones(1024)
        
        # Create datasets
        im = fh.create_group("index_map")
        im.create_dataset("time", data=time)
        im.create_dataset("pol", data=pol)
        im.create_dataset("freq", data=freq)
        im.create_dataset("sinza", data=sinza)
        fh.create_dataset(
            "ringmap",
            shape=(len(pol), len(freq), len(sinza), len(time)),
            dtype=np.float32,
            compression=comp,
            compression_opts=comp_opts,
        )
        fh.create_dataset(
            "weight",
            shape=(len(pol), len(freq), len(time)),
            dtype=np.float32,
            compression=comp,
            compression_opts=comp_opts,
        )
        fh["ringmap"].attrs["axes"] = ("pol", "freq", "sinza", "time")
        fh["weight"].attrs["axes"] = ("pol", "freq", "time")
