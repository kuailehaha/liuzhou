from tools.select_phase3_pilot import choose_pilot


def test_direct_winner_selected_without_worst_anchor_regression():
    assert choose_pilot(
        direct_score_16=0.56,
        worst_anchor_16=0.50,
        worst_anchor_32=0.51,
        average_anchor_16=0.55,
        average_anchor_32=0.56,
    )[0] == 16


def test_direct_winner_rejected_for_large_worst_anchor_regression():
    selected, reason = choose_pilot(
        direct_score_16=0.56,
        worst_anchor_16=0.47,
        worst_anchor_32=0.51,
        average_anchor_16=0.57,
        average_anchor_32=0.55,
    )
    assert selected == 32
    assert "worst-anchor" in reason


def test_close_results_default_to_higher_throughput_16_sim():
    selected, reason = choose_pilot(
        direct_score_16=0.52,
        worst_anchor_16=0.50,
        worst_anchor_32=0.51,
        average_anchor_16=0.55,
        average_anchor_32=0.56,
    )
    assert selected == 16
    assert "higher-throughput" in reason


def test_unhealthy_pilot_is_eliminated():
    assert choose_pilot(
        direct_score_16=0.6,
        worst_anchor_16=0.6,
        worst_anchor_32=0.4,
        average_anchor_16=0.6,
        average_anchor_32=0.4,
        healthy_16=False,
        healthy_32=True,
    )[0] == 32
