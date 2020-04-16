from .kcl_util import *

def load(context, filepath):
    bytestr = None
    with open(filepath, "rb") as f:
        bytestr = f.read()

    octree = Octree.import_(bytestr)

    return {"FINISHED"}
