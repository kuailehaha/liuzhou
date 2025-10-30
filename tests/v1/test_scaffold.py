import importlib


def test_v1_modules_importable():
    """
    Smoke test to ensure the tensorized scaffold packages can be imported.
    This protects against missing __init__ files or syntax errors while the
    actual implementations are still WIP.
    """

    modules = [
        "v1",
        "v1.common",
        "v1.common.tensor_utils",
        "v1.game",
        "v1.game.state_batch",
        "v1.game.rules_tensor",
        "v1.game.move_encoder",
        "v1.net",
        "v1.net.encoding",
        "v1.net.policy_decoder",
        "v1.mcts",
        "v1.mcts.vectorized_mcts",
        "v1.mcts.node_storage",
        "v1.self_play",
        "v1.self_play.runner",
        "v1.self_play.samples",
        "v1.train",
        "v1.train.pipeline",
        "v1.train.dataset",
    ]

    for name in modules:
        importlib.import_module(name)

