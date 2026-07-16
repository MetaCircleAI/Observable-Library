import observable_library as ol


def observable_ids(observables: list[ol.Observable]) -> list[str]:
    return [observable.spec.id for observable in observables]


def runtime_version() -> str:
    return ol.__version__
