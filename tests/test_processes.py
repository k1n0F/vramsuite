from vramsuite.backends.windows_gpu import parse_instance
from vramsuite.core.processes import aggregate_counters

NVIDIA_LUID = (0x0, 0x16A7B)
IGPU_LUID = (0x0, 0x1822D)
PCI = {NVIDIA_LUID: (1, 0, 0), IGPU_LUID: (0, 2, 0)}


def test_instance_names_parse():
    assert parse_instance("pid_44968_luid_0x00000000_0x00016A7B_phys_0") == (44968, 0, 0x16A7B)
    assert parse_instance("luid_0x00000000_0x00016A7B_phys_0") == (None, 0, 0x16A7B)
    assert parse_instance("_Total") is None


def test_only_target_gpu_is_summed_per_pid():
    dedicated = {
        "pid_10_luid_0x00000000_0x00016A7B_phys_0": 100,
        "pid_10_luid_0x00000000_0x00016A7B_phys_1": 50,  # second engine, same GPU
        "pid_10_luid_0x00000000_0x0001822D_phys_0": 999,  # iGPU: dropped
        "pid_20_luid_0x00000000_0x0001822D_phys_0": 7,  # only on iGPU: dropped
        "luid_0x00000000_0x00016A7B_phys_0": 12345,  # adapter-level instance: dropped
    }
    shared = {"pid_10_luid_0x00000000_0x00016A7B_phys_0": 5}

    result = aggregate_counters(
        dedicated,
        shared,
        parse_instance=parse_instance,
        luid_pci=lambda high, low: PCI.get((high, low)),
        target_bus_device=(1, 0),
    )

    assert result == {10: {"dedicated": 150, "shared": 5}}
