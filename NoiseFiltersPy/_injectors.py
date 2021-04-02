from NoiseFiltersPy import RandomInjector, NonlinearwiseInjector, NeighborwiseInjector

_implemented_injectors = {
    "RandomInjector": RandomInjector.RandomInjector,
    "NonlinearwiseInjector": NonlinearwiseInjector.NonlinearwiseInjector,
    "NeighborwiseInjector": NeighborwiseInjector.NeighborwiseInjector
}