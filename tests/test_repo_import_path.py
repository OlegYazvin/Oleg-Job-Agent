from pathlib import Path


def test_repo_root_job_agent_namespace_points_to_src_tree() -> None:
    import job_agent

    expected = Path(__file__).resolve().parent.parent / "src" / "job_agent"
    assert str(expected) in list(job_agent.__path__)
