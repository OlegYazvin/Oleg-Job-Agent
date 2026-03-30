from job_agent.dashboard import _compute_dashboard_minimum_size


def test_compute_dashboard_minimum_size_respects_defaults() -> None:
    assert _compute_dashboard_minimum_size(900, 600) == (1080, 720)


def test_compute_dashboard_minimum_size_expands_for_required_layout() -> None:
    assert _compute_dashboard_minimum_size(1420, 830) == (1420, 830)
