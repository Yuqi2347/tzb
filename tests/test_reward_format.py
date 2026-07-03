import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from remote_sensing_mllm.compute_reward import compute_format_reward


def test_format_reward_accepts_think_and_answer_tags():
    score = compute_format_reward("<think>reason</think><answer>answer</answer>")
    assert score > 0


def test_format_reward_penalizes_missing_tags():
    good = compute_format_reward("<think>reason</think><answer>answer</answer>")
    bad = compute_format_reward("answer only")
    assert good > bad
