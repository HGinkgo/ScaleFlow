from importlib import import_module


def test_scaleflow_can_be_imported() -> None:
    scaleflow = import_module("scaleflow")

    assert getattr(scaleflow, "__version__", None) == "0.1.0"
