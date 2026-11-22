from maxent.mcmc.sampler import DualAveragingStepAdapter


def test_dual_averaging_moves_step_toward_target():
    high_acc = DualAveragingStepAdapter(
        step=0.05,
        target_accept=0.50,
        step_min=1e-4,
        step_max=1.0,
    )
    low_acc = DualAveragingStepAdapter(
        step=0.05,
        target_accept=0.50,
        step_min=1e-4,
        step_max=1.0,
    )

    for _ in range(12):
        high_acc.update(0.90)
        low_acc.update(0.10)

    assert high_acc.step > high_acc.step_init
    assert low_acc.step < low_acc.step_init


def test_dual_averaging_freeze_and_clamp():
    adapter = DualAveragingStepAdapter(
        step=1e-12,
        target_accept=0.80,
        step_min=1e-3,
        step_max=2e-3,
    )
    # Invalid small init is clamped to min.
    assert adapter.step == 1e-3

    for _ in range(30):
        adapter.update(1.0)
    assert 1e-3 <= adapter.step <= 2e-3

    frozen = adapter.freeze()
    after_freeze = adapter.update(0.0)
    assert frozen == after_freeze
    assert adapter.frozen
