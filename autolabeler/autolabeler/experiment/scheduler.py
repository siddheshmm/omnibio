"""Randomized block scheduler for experiment trial generation.

Generates a balanced, randomized trial plan from an ExperimentConfig.
Each 'block' contains every class exactly once in shuffled order,
ensuring balanced representation across time.
"""

import random
from dataclasses import dataclass
from typing import Optional

from autolabeler.config import ExperimentConfig


@dataclass(frozen=True)
class Trial:
    """A single trial in the experiment schedule.

    Attributes:
        trial_number: 1-indexed trial number (overall).
        class_label: The class name for this trial (e.g. "Touch", "No Touch").
        block_number: 1-indexed block this trial belongs to.
        index_in_block: 0-indexed position within the block.
    """
    trial_number: int
    class_label: str
    block_number: int
    index_in_block: int


def generate_schedule(
    config: ExperimentConfig,
    seed: Optional[int] = None,
) -> list[Trial]:
    """Generate the full experiment trial schedule.

    Creates randomized blocks where each block contains every class
    exactly once in shuffled order. Total trials = trials_per_class × len(classes).

    Args:
        config: Experiment configuration with classes, trials_per_class, etc.
        seed: Optional random seed for reproducibility.

    Returns:
        Ordered list of Trial objects representing the full experiment.

    Raises:
        ValueError: If config has no classes or invalid trials_per_class.
    """
    classes = config.classes
    if not classes:
        raise ValueError("Experiment must have at least one class.")
    if config.trials_per_class < 1:
        raise ValueError("trials_per_class must be >= 1.")

    rng = random.Random(seed)
    block_size = config.effective_block_size
    num_blocks = config.trials_per_class  # each block has each class once

    schedule: list[Trial] = []
    trial_num = 1

    for block_idx in range(num_blocks):
        # Build one block: each class appears once
        block_classes = list(classes)
        if config.randomize:
            rng.shuffle(block_classes)

        for pos, cls in enumerate(block_classes):
            schedule.append(Trial(
                trial_number=trial_num,
                class_label=cls,
                block_number=block_idx + 1,
                index_in_block=pos,
            ))
            trial_num += 1

    return schedule


def schedule_summary(schedule: list[Trial], config: ExperimentConfig) -> dict:
    """Generate a summary of the schedule for display.

    Returns:
        Dict with keys: total_trials, num_blocks, classes, trials_per_class,
        estimated_duration_seconds, estimated_duration_str.
    """
    total = len(schedule)
    num_blocks = schedule[-1].block_number if schedule else 0
    est_secs = config.estimated_duration

    mins, secs = divmod(int(est_secs), 60)
    duration_str = f"{mins}m {secs}s" if mins else f"{secs}s"

    return {
        "total_trials": total,
        "num_blocks": num_blocks,
        "classes": list(config.classes),
        "trials_per_class": config.trials_per_class,
        "trial_duration": config.trial_duration,
        "rest_duration": config.rest_duration,
        "estimated_duration_seconds": est_secs,
        "estimated_duration_str": duration_str,
    }
