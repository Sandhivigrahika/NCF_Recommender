import h5py, json

with h5py.File("../model/ncf_model.h5", "r") as f:
    print("keras_version:" , f.attrs.get("keras_version"))
    print("backend:", f.attrs.get("backend"))
