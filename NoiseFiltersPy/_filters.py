from NoiseFiltersPy import AENN, CNN, DROP, ENN, TomekLinks

_implemented_filters = {
    "AENN": AENN.AENN,
    "CNN": CNN.CNN,
    "DROP": DROP.DROPv1,
    "ENN": ENN.ENN,
    "TomekLinks": TomekLinks.TomekLinks
}