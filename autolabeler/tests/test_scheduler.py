"""Tests for the experiment scheduler module."""

from autolabeler.config import ExperimentConfig
from autolabeler.experiment.scheduler import generate_schedule, schedule_summary, Trial


class TestScheduleGeneration:
    """Test randomized block schedule generation."""

    def test_basic_two_class_schedule(self):
        config = ExperimentConfig(classes=["Touch", "No Touch"], trials_per_class=5)
        schedule = generate_schedule(config, seed=42)

        assert len(schedule) == 10  # 2 classes × 5 trials each
        assert all(isinstance(t, Trial) for t in schedule)

    def test_trial_numbers_are_sequential(self):
        config = ExperimentConfig(classes=["A", "B", "C"], trials_per_class=4)
        schedule = generate_schedule(config, seed=42)

        trial_nums = [t.trial_number for t in schedule]
        assert trial_nums == list(range(1, 13))  # 3×4 = 12 trials

    def test_each_block_contains_all_classes(self):
        classes = ["Touch", "No Touch", "Pressure"]
        config = ExperimentConfig(classes=classes, trials_per_class=10)
        schedule = generate_schedule(config, seed=42)

        # Group by block
        blocks: dict[int, list[str]] = {}
        for t in schedule:
            blocks.setdefault(t.block_number, []).append(t.class_label)

        # Each block should have exactly one of each class
        for block_num, labels in blocks.items():
            assert sorted(labels) == sorted(classes), (
                f"Block {block_num} has {labels}, expected one of each class"
            )

    def test_class_counts_are_balanced(self):
        config = ExperimentConfig(
            classes=["A", "B", "C"], trials_per_class=20
        )
        schedule = generate_schedule(config, seed=42)

        counts = {}
        for t in schedule:
            counts[t.class_label] = counts.get(t.class_label, 0) + 1

        assert counts == {"A": 20, "B": 20, "C": 20}

    def test_randomized_order(self):
        config = ExperimentConfig(
            classes=["A", "B", "C"], trials_per_class=10, randomize=True
        )
        schedule = generate_schedule(config, seed=42)

        # With randomization, at least some block should not be in sorted order
        blocks: dict[int, list[str]] = {}
        for t in schedule:
            blocks.setdefault(t.block_number, []).append(t.class_label)

        has_shuffled = any(
            labels != sorted(labels) for labels in blocks.values()
        )
        assert has_shuffled, "Expected at least one shuffled block"

    def test_non_randomized_is_sequential(self):
        config = ExperimentConfig(
            classes=["A", "B", "C"], trials_per_class=3, randomize=False
        )
        schedule = generate_schedule(config, seed=42)

        # Without randomization, every block should be [A, B, C]
        blocks: dict[int, list[str]] = {}
        for t in schedule:
            blocks.setdefault(t.block_number, []).append(t.class_label)

        for block_num, labels in blocks.items():
            assert labels == ["A", "B", "C"], (
                f"Block {block_num}: expected ['A', 'B', 'C'], got {labels}"
            )

    def test_single_class(self):
        config = ExperimentConfig(classes=["Only"], trials_per_class=5)
        schedule = generate_schedule(config, seed=42)

        assert len(schedule) == 5
        assert all(t.class_label == "Only" for t in schedule)

    def test_seed_produces_reproducible_results(self):
        config = ExperimentConfig(
            classes=["X", "Y", "Z"], trials_per_class=10
        )
        s1 = generate_schedule(config, seed=123)
        s2 = generate_schedule(config, seed=123)

        labels1 = [t.class_label for t in s1]
        labels2 = [t.class_label for t in s2]
        assert labels1 == labels2

    def test_different_seeds_produce_different_results(self):
        config = ExperimentConfig(
            classes=["X", "Y", "Z"], trials_per_class=10
        )
        s1 = generate_schedule(config, seed=1)
        s2 = generate_schedule(config, seed=2)

        labels1 = [t.class_label for t in s1]
        labels2 = [t.class_label for t in s2]
        assert labels1 != labels2

    def test_block_numbers_are_sequential(self):
        config = ExperimentConfig(classes=["A", "B"], trials_per_class=5)
        schedule = generate_schedule(config, seed=42)

        blocks = [t.block_number for t in schedule]
        # Should be [1,1, 2,2, 3,3, 4,4, 5,5]
        expected = []
        for b in range(1, 6):
            expected.extend([b, b])
        assert blocks == expected

    def test_empty_classes_raises(self):
        config = ExperimentConfig(classes=[], trials_per_class=5)
        try:
            generate_schedule(config, seed=42)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "at least one class" in str(e)

    def test_zero_trials_raises(self):
        config = ExperimentConfig(classes=["A"], trials_per_class=0)
        try:
            generate_schedule(config, seed=42)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "trials_per_class" in str(e)


class TestScheduleSummary:
    """Test schedule summary generation."""

    def test_summary_fields(self):
        config = ExperimentConfig(
            classes=["Touch", "No Touch"],
            trials_per_class=10,
            trial_duration=5.0,
            rest_duration=3.0,
        )
        schedule = generate_schedule(config, seed=42)
        summary = schedule_summary(schedule, config)

        assert summary["total_trials"] == 20
        assert summary["num_blocks"] == 10
        assert summary["classes"] == ["Touch", "No Touch"]
        assert summary["trials_per_class"] == 10
        assert summary["trial_duration"] == 5.0
        assert summary["rest_duration"] == 3.0
        assert summary["estimated_duration_seconds"] > 0
        assert "m" in summary["estimated_duration_str"] or "s" in summary["estimated_duration_str"]

    def test_estimated_duration(self):
        config = ExperimentConfig(
            classes=["A", "B"],
            trials_per_class=5,
            trial_duration=2.0,
            rest_duration=1.0,
            countdown_duration=3.0,
        )
        schedule = generate_schedule(config, seed=42)
        summary = schedule_summary(schedule, config)

        # 10 trials × (2s + 1s) + 3s countdown = 33s
        assert summary["estimated_duration_seconds"] == 33.0
