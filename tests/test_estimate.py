from vramsuite.core.estimate import offload_seconds_per_step, plan_memory


def test_fits_with_headroom():
    plan = plan_memory(weights_mb=8000, activations_mb=2000, available_mb=14000, reserve_mb=1000)
    assert plan["status"] == "fits"
    assert plan["budget_mb"] == 13000
    assert plan["headroom_mb"] == 3000
    assert plan["offload_mb"] == 0


def test_overflow_is_offloaded_from_weights():
    plan = plan_memory(weights_mb=10000, activations_mb=4000, available_mb=12000, reserve_mb=1000)
    assert plan["status"] == "offload"
    assert plan["overflow_mb"] == 3000
    assert plan["offload_mb"] == 3000


def test_activations_alone_too_big():
    plan = plan_memory(weights_mb=1000, activations_mb=12000, available_mb=12000, reserve_mb=1000)
    assert plan["status"] == "does_not_fit"
    assert plan["offload_mb"] == 0


def test_offload_cost():
    # 1000 MiB over 28.5 GB/s pinned PCIe
    assert offload_seconds_per_step(1000, 28.5) == 0.037
    assert offload_seconds_per_step(1000, None) is None
