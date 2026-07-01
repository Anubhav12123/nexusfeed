from nexusfeed.experiments.bandit import EpsilonGreedyBandit


def test_bandit_exploits_best_arm_when_epsilon_zero():
    bandit = EpsilonGreedyBandit(epsilon=0.0)
    bandit.record_reward("item_a", 1.0)
    bandit.record_reward("item_a", 1.0)
    bandit.record_reward("item_b", 0.0)

    selected = bandit.select_exploration_slot(["item_a", "item_b"])
    assert selected == "item_a"


def test_bandit_explores_when_epsilon_one():
    bandit = EpsilonGreedyBandit(epsilon=1.0)
    bandit.record_reward("item_a", 1.0)
    bandit.record_reward("item_b", 0.0)

    # With epsilon=1 every call is exploration (random), so over many trials
    # both arms should get selected at least once.
    selections = {bandit.select_exploration_slot(["item_a", "item_b"]) for _ in range(200)}
    assert selections == {"item_a", "item_b"}


def test_inject_exploration_slots_preserves_length():
    bandit = EpsilonGreedyBandit(epsilon=0.5)
    feed = [f"item_{i}" for i in range(20)]
    candidates = feed + [f"extra_{i}" for i in range(30)]
    result = bandit.inject_exploration_slots(feed, candidates)
    assert len(result) == len(feed)
