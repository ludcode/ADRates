import importlib.util
import os

# Load module directly from file path because package lacks __init__
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cavour', 'utils', 'global_types.py'))
spec = importlib.util.spec_from_file_location('global_types', module_path)
gt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gt)


def test_swap_types_enum():
    assert gt.SwapTypes.PAY.name == "PAY"
    assert gt.SwapTypes.RECEIVE.value == 2


def test_request_types_enum():
    assert gt.RequestTypes.VALUE.value == 1
    assert gt.RequestTypes.GAMMA.name == "GAMMA"