import pytest

from src.rewards import get_reward_function
from src.rewards.file_localization.file_localization import (
    atomic_outcome_localization_reward,
    gold_file_read_process_reward,
    multilevel_localization_f1_reward,
)


def _instance(file_changes):
    return {"file_changes": file_changes}


def _method_change():
    return _instance(
        [
            {
                "file": "pkg/foo.py",
                "changes": {
                    "edited_modules": ["pkg/foo.py:Widget"],
                    "edited_entities": ["pkg/foo.py:Widget.render"],
                },
            }
        ]
    )


def test_atomic_outcome_exact_tuple_match():
    reward, metrics = atomic_outcome_localization_reward(
        final_message="",
        instance=_method_change(),
        structured_locations=[
            {"file": "pkg/foo.py", "class_name": "Widget", "function_name": "render"}
        ],
    )

    assert reward == pytest.approx(1.0)
    assert metrics["atomic_file_reward"] == pytest.approx(1.0)
    assert metrics["atomic_class_reward"] == pytest.approx(1.0)
    assert metrics["atomic_function_reward"] == pytest.approx(1.0)
    assert metrics["atomic_tuple_reward"] == pytest.approx(1.0)


def test_atomic_outcome_file_correct_function_wrong():
    reward, metrics = atomic_outcome_localization_reward(
        final_message="",
        instance=_method_change(),
        structured_locations=[
            {"file": "pkg/foo.py", "class_name": "Widget", "function_name": "close"}
        ],
    )

    assert metrics["atomic_file_reward"] == pytest.approx(1.0)
    assert metrics["atomic_class_reward"] == pytest.approx(1.0)
    assert metrics["atomic_function_reward"] == pytest.approx(0.0)
    assert metrics["atomic_tuple_reward"] == pytest.approx(0.0)
    assert reward == pytest.approx(0.35)


def test_atomic_outcome_function_name_correct_but_wrong_binding():
    reward, metrics = atomic_outcome_localization_reward(
        final_message="",
        instance=_method_change(),
        structured_locations=[
            {"file": "pkg/bar.py", "class_name": "Other", "function_name": "render"}
        ],
    )

    assert metrics["atomic_file_reward"] == pytest.approx(0.0)
    assert metrics["atomic_class_reward"] == pytest.approx(0.0)
    assert metrics["atomic_function_reward"] == pytest.approx(1.0)
    assert metrics["atomic_tuple_reward"] == pytest.approx(0.0)
    assert reward == pytest.approx(0.45)


def test_atomic_outcome_top_level_function_match():
    reward, metrics = atomic_outcome_localization_reward(
        final_message="",
        instance=_instance(
            [
                {
                    "file": "pkg/foo.py",
                    "changes": {"edited_modules": None, "edited_entities": ["pkg/foo.py:parse"]},
                }
            ]
        ),
        structured_locations=[
            {"file": "pkg/foo.py", "class_name": None, "function_name": "parse"}
        ],
    )

    assert reward == pytest.approx(1.0)
    assert metrics["atomic_class_reward"] == pytest.approx(0.0)
    assert metrics["atomic_function_reward"] == pytest.approx(1.0)


def test_atomic_outcome_file_only_match():
    reward, metrics = atomic_outcome_localization_reward(
        final_message="",
        instance=_instance([{"file": "pkg/settings.py", "changes": {}}]),
        structured_locations=[
            {"file": "pkg/settings.py", "class_name": None, "function_name": None}
        ],
    )

    assert reward == pytest.approx(1.0)
    assert metrics["atomic_file_reward"] == pytest.approx(1.0)
    assert metrics["atomic_tuple_reward"] == pytest.approx(1.0)


def test_atomic_outcome_requires_structured_locations():
    reward, metrics = atomic_outcome_localization_reward(
        final_message="",
        instance=_method_change(),
        structured_locations=None,
    )

    assert reward == 0
    assert metrics["atomic_outcome_localization_reward"] == 0


def test_gold_file_read_process_cat_rewards_once():
    reward, metrics = gold_file_read_process_reward(
        messages=[
            {"kind": "ActionEvent", "action": {"command": "cat pkg/foo.py"}},
            {"kind": "ActionEvent", "action": {"command": "cat pkg/foo.py"}},
        ],
        instance=_method_change(),
    )

    assert reward == pytest.approx(1.0)
    assert metrics["gold_files_opened_count"] == 1
    assert metrics["gold_files_total"] == 1


def test_gold_file_read_process_sed_rewards():
    reward, metrics = gold_file_read_process_reward(
        messages=[
            {"kind": "ActionEvent", "action": {"command": "sed -n '1,80p' pkg/foo.py"}}
        ],
        instance=_method_change(),
    )

    assert reward == pytest.approx(1.0)
    assert metrics["gold_file_open_coverage"] == pytest.approx(1.0)


def test_gold_file_read_process_does_not_reward_rg_path_output():
    reward, metrics = gold_file_read_process_reward(
        messages=[
            {"kind": "ActionEvent", "action": {"command": "rg render pkg/foo.py"}}
        ],
        instance=_method_change(),
    )

    assert reward == pytest.approx(0.0)
    assert metrics["gold_files_opened_count"] == 0


def test_gold_file_read_process_multiple_gold_files_normalized():
    reward, metrics = gold_file_read_process_reward(
        messages=[
            {"kind": "ActionEvent", "action": {"command": "cat pkg/foo.py"}}
        ],
        instance=_instance(
            [
                {"file": "pkg/foo.py", "changes": {"edited_entities": ["pkg/foo.py:parse"]}},
                {"file": "pkg/bar.py", "changes": {"edited_entities": ["pkg/bar.py:format"]}},
            ]
        ),
    )

    assert reward == pytest.approx(0.5)
    assert metrics["gold_files_opened_count"] == 1
    assert metrics["gold_files_total"] == 2


def test_reward_registry_includes_new_rewards_and_old_reward_still_registered():
    assert get_reward_function("atomic_outcome_localization_reward") is atomic_outcome_localization_reward
    assert get_reward_function("gold_file_read_process_reward") is gold_file_read_process_reward
    assert get_reward_function("multilevel_localization_f1_reward") is multilevel_localization_f1_reward


def test_new_reward_configs_load():
    omegaconf = pytest.importorskip("omegaconf")
    config_names = [
        "configs/reward_old_continue.yaml",
        "configs/reward_atomic_outcome.yaml",
        "configs/reward_atomic_outcome_process_v1.yaml",
    ]

    for config_name in config_names:
        cfg = omegaconf.OmegaConf.load(config_name)
        assert len(cfg.reward) >= 1
        assert all("fn" in item for item in cfg.reward)
        assert "terminal" in cfg.tools
