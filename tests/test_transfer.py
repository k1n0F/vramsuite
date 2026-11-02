from vramsuite.core.transfer import pcie_theoretical_gb_s


def test_pcie_theoretical_bandwidth():
    assert pcie_theoretical_gb_s(5, 8) == 31.5  # RTX 5080 in a shared x8 slot
    assert pcie_theoretical_gb_s(5, 16) == 63.0
    assert pcie_theoretical_gb_s(3, 16) == 15.8
    assert pcie_theoretical_gb_s(2, 16) == 8.0  # 8b/10b encoding
    assert pcie_theoretical_gb_s(None, 8) is None
    assert pcie_theoretical_gb_s(5, None) is None
