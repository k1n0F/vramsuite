from vramsuite.core.vramcard import create_vramcard


def _nvml_device(index, name, bus, cc):
    return {
        "index": index,
        "name": name,
        "total_vram_mb": 1000 * (index + 1),
        "free_vram_mb": 500,
        "used_vram_mb": 500,
        "details": {"compute_capability": cc, "pci": {"domain": 0, "bus": bus, "device": 0}},
    }


def _fingerprint(cuda_bus):
    return {
        "runtime": {},
        "torch": {
            "cuda_available": True,
            "devices": [{"compute_capability": "9.9", "major": 9, "minor": 9,
                         "multi_processor_count": 1}],
        },
        "nvml": {
            "available": True,
            "devices": [
                _nvml_device(0, "GPU on bus 1", bus=1, cc="8.9"),
                _nvml_device(1, "GPU on bus 2", bus=2, cc="12.0"),
            ],
        },
        "cuda_runtime": {
            "current_device": 0,
            "attributes": {
                "pci_domain_id": 0, "pci_bus_id": cuda_bus, "pci_device_id": 0,
                "compute_capability_major": 12, "compute_capability_minor": 0,
                "multiprocessor_count": 84,
            },
        },
    }


def test_cuda_device_is_matched_to_nvml_by_pci_not_index():
    card = create_vramcard(_fingerprint(cuda_bus=2))

    assert card["device"]["nvml_match"] == "pci_bus_id"
    assert card["device"]["nvml_device_index"] == 1
    assert card["gpu"]["name"] == "GPU on bus 2"
    assert card["memory"]["driver_total_mb"] == 2000


def test_nvml_wins_over_native_and_torch_only_fills_gaps():
    gpu = create_vramcard(_fingerprint(cuda_bus=2))["gpu"]

    assert gpu["compute_capability"] == "12.0"  # NVML, not torch "9.9"
    assert gpu["multi_processor_count"] == 84  # native CUDA, not torch 1
    assert gpu["source"] == "nvml"
