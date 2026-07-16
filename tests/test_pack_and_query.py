from __future__ import annotations

import observable_library as ol


def _observable() -> ol.Observable:
    spec = ol.ObservableSpec(
        source="param.layer.weight", selector="all", reduction="sum"
    )
    return ol.Observable(
        spec=spec, compute=lambda tensors, context: tensors[spec.source]
    )


def test_pack_is_a_stable_runtime_container() -> None:
    observable = _observable()
    pack = ol.Pack([observable])

    runtime = ol.Runtime(pack, tensors={observable.spec.source: 3})

    assert len(pack) == 1
    assert list(pack) == [observable]
    assert runtime.observe(step=0) == {observable.spec.id: 3}


def test_query_reads_local_storage_by_observable_id_and_step(tmp_path) -> None:
    storage = ol.LocalStorage(tmp_path)
    storage("loss", 4, 1.5, {"source": "loss"})

    assert ol.query(storage, observable_id="loss", step=4) == 1.5
