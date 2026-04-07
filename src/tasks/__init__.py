"""Task registry for OpenEnv support environment."""

from src.tasks.easy import get_scenario as get_easy_scenario
from src.tasks.medium import get_scenario as get_medium_scenario
from src.tasks.hard import get_scenario as get_hard_scenario

TASK_REGISTRY = {
    "easy": get_easy_scenario,
    "medium": get_medium_scenario,
    "hard": get_hard_scenario,
}


def get_task(task_name: str):
    """Retrieve a task scenario by difficulty name."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_name]()
