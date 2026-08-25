import pytest

import hardware_guard as hg


@pytest.fixture()
def fake_torch(monkeypatch):
    class FakeProps:
        total_memory = 11_000_000_000
        driver_version = "575.51"

    class FakeTorch:
        class cuda:  # noqa: N801 - mirrors torch namespace
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def get_device_name(i):
                return FakeTorch._name

            @staticmethod
            def get_device_properties(i):
                return FakeProps()

    FakeTorch._name = "NVIDIA GeForce RTX 2080 Ti"
    monkeypatch.setitem(__import__("sys").modules, "torch", FakeTorch)
    return FakeTorch


def test_assert_gpu_passes_on_locked_card(fake_torch):
    info = hg.assert_gpu()
    assert info["gpu_name"] == "NVIDIA GeForce RTX 2080 Ti"
    assert info["driver_version"] == "575.51"


def test_assert_gpu_aborts_on_drifted_card(fake_torch, monkeypatch):
    import socket

    fake_torch._name = "NVIDIA GeForce RTX 3080"  # the observed gnode077 swap
    with pytest.raises(RuntimeError, match=socket.gethostname()):
        hg.assert_gpu()


def test_assert_gpu_aborts_when_no_gpu(fake_torch, monkeypatch):
    fake_torch.cuda.is_available = staticmethod(lambda: False)
    # get_device_name still returns a name but cuda unavailable -> guard must abort
    with pytest.raises(RuntimeError, match="expected name containing"):
        hg.assert_gpu()


def test_gpu_info_never_raises(monkeypatch):
    # broken torch import path -> dict with error key, no exception
    monkeypatch.setitem(
        __import__("sys").modules, "torch",
        __import__("builtins"),  # not a torch-like object; getattr chain explodes
    )
    info = hg.gpu_info()
    assert "error" in info or info["gpu_name"] is None
