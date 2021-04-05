from NoiseFiltersPy import AENN, HARF, CNN, DROP, ENN, TomekLinks

_implemented_filters = {
    "HARF": HARF.HARF,
    "CNN": CNN.CNN,
    "DROP": DROP.DROPv1,
    "ENN": ENN.ENN,
    "AENN": AENN.AENN,
    "TomekLinks": TomekLinks.TomekLinks
}