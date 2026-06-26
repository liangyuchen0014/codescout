import ast
import json
import shlex
from collections.abc import Iterable

from .module_rewards import get_simple_results_from_raw_outputs, parse_structured_outputs

from src.rewards import reward


def _iter_values(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _normalize_optional_name(value):
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _normalize_file_path(path):
    if path is None:
        return None
    path = str(path).strip().strip("'\"")
    if not path:
        return None
    path = path.replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path.strip("/")


def _parse_location_identifier(identifier):
    identifier = str(identifier).strip()
    if ":" not in identifier:
        return _normalize_file_path(identifier), None, None

    file_path, symbol = identifier.split(":", 1)
    file_path = _normalize_file_path(file_path)
    symbol = symbol.strip()
    if not symbol:
        return file_path, None, None

    if "." in symbol:
        class_name, function_name = symbol.rsplit(".", 1)
        return file_path, _normalize_optional_name(class_name), _normalize_optional_name(function_name)

    return file_path, None, _normalize_optional_name(symbol)


def _looks_like_class_name(name):
    return bool(name) and name[0].isupper()


def _location_tuple(file_path, class_name=None, function_name=None):
    normalized_file = _normalize_file_path(file_path)
    if normalized_file is None:
        return None
    return (
        normalized_file,
        _normalize_optional_name(class_name),
        _normalize_optional_name(function_name),
    )


def _sets_from_locations(locations):
    files = set()
    classes = set()
    functions = set()
    tuples = set()

    for location in locations:
        if location is None:
            continue
        file_path, class_name, function_name = location
        files.add(file_path)
        if class_name is not None:
            classes.add(class_name)
        if function_name is not None:
            functions.add(function_name)
        tuples.add(location)

    return files, classes, functions, tuples


def _gold_atomic_locations(instance):
    locations = set()

    for change in instance.get("file_changes", []):
        file_path = _normalize_file_path(change.get("file"))
        if file_path is None:
            continue

        module_locations = set()
        entity_locations = set()
        changes = change.get("changes") or {}

        for module in _iter_values(changes.get("edited_modules")):
            module_file, class_name, function_name = _parse_location_identifier(module)
            if module_file is None:
                module_file = file_path
            if class_name is not None:
                module_locations.add(_location_tuple(module_file, class_name, None))
            elif function_name is not None:
                if _looks_like_class_name(function_name):
                    module_locations.add(_location_tuple(module_file, function_name, None))
                else:
                    module_locations.add(_location_tuple(module_file, None, function_name))

        for entity in _iter_values(changes.get("edited_entities")):
            entity_locations.add(_location_tuple(*_parse_location_identifier(entity)))

        if entity_locations:
            covered_modules = {
                (entity_file, entity_class, entity_function)
                for entity_file, entity_class, entity_function in entity_locations
            }
            covered_class_modules = {
                (entity_file, entity_class, None)
                for entity_file, entity_class, _ in entity_locations
                if entity_class is not None
            }
            change_locations = set(entity_locations)
            change_locations.update(
                module_location
                for module_location in module_locations
                if module_location not in covered_modules and module_location not in covered_class_modules
            )
        else:
            change_locations = set(module_locations)

        if not change_locations:
            change_locations.add(_location_tuple(file_path, None, None))

        locations.update(loc for loc in change_locations if loc is not None)

    return locations


def _predicted_atomic_locations(structured_locations):
    locations = set()
    found_empty_filename = False

    for location in structured_locations or []:
        tuple_location = _location_tuple(
            location.get("file"),
            location.get("class_name"),
            location.get("function_name"),
        )
        if tuple_location is None:
            found_empty_filename = True
            break
        locations.add(tuple_location)

    if found_empty_filename:
        return set()
    return locations


def _extract_command_from_action(action):
    if not isinstance(action, dict):
        return None

    if isinstance(action.get("command"), str):
        return action["command"]

    for value in action.values():
        if isinstance(value, dict):
            command = _extract_command_from_action(value)
            if command is not None:
                return command
        elif isinstance(value, str) and "command" in value:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                continue
            command = _extract_command_from_action(parsed)
            if command is not None:
                return command
    return None


def _extract_terminal_commands(messages):
    commands = []
    for message in messages or []:
        if message.get("kind") == "ActionEvent":
            command = _extract_command_from_action(message.get("action", {}))
            if command:
                commands.append(command)
            continue

        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls", []):
            function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                continue
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
            command = parsed_arguments.get("command")
            if isinstance(command, str):
                commands.append(command)
    return commands


def _candidate_read_paths(command):
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()

    if not tokens:
        return []

    executable = tokens[0].rsplit("/", 1)[-1]
    if executable == "cat":
        return [
            token
            for token in tokens[1:]
            if not token.startswith("-") and not token.startswith(">") and token.endswith(".py")
        ]

    if executable == "sed":
        return [
            token
            for token in tokens[1:]
            if not token.startswith("-") and token.endswith(".py")
        ]

    return []


def _path_matches_gold(candidate_path, gold_file):
    candidate = _normalize_file_path(candidate_path)
    gold = _normalize_file_path(gold_file)
    if candidate is None or gold is None:
        return False
    return candidate == gold or candidate.endswith(f"/{gold}")


def compute_file_f1_score(predicted_files, true_files, beta=1.0):
    pred, true = set(predicted_files), set(true_files)
    if not true:
        return 0.0 # return 0 reward if ground truth is empty
    tp = len(pred & true)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0

# def file_localization_f1_reward(final_message, instance):
#     predicted_files = set(ast.literal_eval(final_message.split("<file-list>")[1].split("</file-list>")[0]))
#     # print("Predicted files:", predicted_files)
#     true_files = set(x[0] for x in ast.literal_eval(instance["target"]))
#     # print("True files:", true_files)
#     return compute_file_f1_score(predicted_files, true_files)

@reward("file_localization_f1_reward")
def file_localization_f1_reward(
    final_message: str,
    instance: dict,
    file_level_weight: float=1.0,
    beta: float=1.0,
    **kwargs
    ):
    all_found_files, all_found_modules, all_found_entities = get_simple_results_from_raw_outputs(final_message)
    true_files = set(x[0] for x in ast.literal_eval(instance["target"]))
    file_level_score = compute_file_f1_score(all_found_files, true_files, beta=beta)
    weighted_file_score = file_level_weight * file_level_score

    return weighted_file_score, {"file_reward": file_level_score}


@reward("atomic_outcome_localization_reward")
def atomic_outcome_localization_reward(
    final_message: str,
    instance: dict,
    structured_locations: list[dict] | None = None,
    file_level_weight: float = 0.20,
    class_level_weight: float = 0.15,
    function_level_weight: float = 0.45,
    tuple_level_weight: float = 0.20,
    **kwargs
):
    if structured_locations is None:
        return 0, {
            "atomic_outcome_localization_reward": 0,
            "atomic_file_reward": 0,
            "atomic_class_reward": 0,
            "atomic_function_reward": 0,
            "atomic_tuple_reward": 0,
        }

    gold_locations = _gold_atomic_locations(instance)
    predicted_locations = _predicted_atomic_locations(structured_locations)

    gold_files, gold_classes, gold_functions, gold_tuples = _sets_from_locations(gold_locations)
    predicted_files, predicted_classes, predicted_functions, predicted_tuples = _sets_from_locations(predicted_locations)

    file_f1_score = compute_file_f1_score(predicted_files, gold_files)
    class_f1_score = compute_file_f1_score(predicted_classes, gold_classes)
    function_f1_score = compute_file_f1_score(predicted_functions, gold_functions)
    tuple_f1_score = compute_file_f1_score(predicted_tuples, gold_tuples)

    weighted_components = [
        (file_f1_score, file_level_weight, gold_files),
        (class_f1_score, class_level_weight, gold_classes),
        (function_f1_score, function_level_weight, gold_functions),
        (tuple_f1_score, tuple_level_weight, gold_tuples),
    ]
    active_weight = sum(weight for _, weight, gold_set in weighted_components if gold_set)
    reward_value = (
        sum(score * weight for score, weight, gold_set in weighted_components if gold_set) / active_weight
        if active_weight > 0
        else 0
    )

    return reward_value, {
        "atomic_outcome_localization_reward": reward_value,
        "atomic_file_reward": file_f1_score,
        "atomic_class_reward": class_f1_score,
        "atomic_function_reward": function_f1_score,
        "atomic_tuple_reward": tuple_f1_score,
    }


@reward("gold_file_read_process_reward")
def gold_file_read_process_reward(
    messages,
    instance: dict,
    **kwargs
):
    gold_files = {
        location[0]
        for location in _gold_atomic_locations(instance)
        if location is not None
    }
    if not gold_files:
        return 0, {
            "gold_file_read_process_reward": 0,
            "gold_files_opened_count": 0,
            "gold_files_total": 0,
            "gold_file_open_coverage": 0,
        }

    opened_gold_files = set()
    for command in _extract_terminal_commands(messages):
        for candidate_path in _candidate_read_paths(command):
            for gold_file in gold_files:
                if _path_matches_gold(candidate_path, gold_file):
                    opened_gold_files.add(gold_file)

    coverage = len(opened_gold_files) / len(gold_files)
    return coverage, {
        "gold_file_read_process_reward": coverage,
        "gold_files_opened_count": len(opened_gold_files),
        "gold_files_total": len(gold_files),
        "gold_file_open_coverage": coverage,
    }


@reward("multilevel_localization_f1_reward")
def multilevel_localization_f1_reward(
    final_message: str,
    instance: dict,
    structured_locations: list[dict] | None = None,
    file_level_weight: float=1.0,
    module_level_weight: float=1.0,
    entity_level_weight: float=1.0,
    **kwargs
    ):

    if structured_locations is None:
        return 0, {
        "multilevel_localization_f1_reward": 0,
        "file_reward": 0,
        "module_reward": 0,
        "entity_reward": 0,
    }

    gt_files = []
    gt_modules = []
    gt_entities = []
    reward = 0

    for change in instance.get("file_changes", []):
        if "file" in change:
            gt_files.append(change["file"])
        if "changes" in change:
            edited_modules = change["changes"].get("edited_modules", [])
            edited_modules = [] if edited_modules is None else edited_modules
            for module in edited_modules:
                gt_modules.append(module)

            edited_entities = change["changes"].get("edited_entities", [])
            edited_entities = [] if edited_entities is None else edited_entities
            for entity in edited_entities:
                gt_entities.append(entity)
    gt_files = set(gt_files)
    gt_modules = set(gt_modules)
    gt_entities = set(gt_entities)

    if structured_locations is not None:
        predicted_files, predicted_modules, predicted_entities = parse_structured_outputs(structured_locations)
    else:
        predicted_files, predicted_modules, predicted_entities = get_simple_results_from_raw_outputs(final_message)

    file_f1_score = compute_file_f1_score(predicted_files, gt_files)
    module_f1_score = compute_file_f1_score(predicted_modules, gt_modules)
    entity_f1_score = compute_file_f1_score(predicted_entities, gt_entities)

    # weight_total = file_level_weight + module_level_weight + entity_level_weight
    # file_level_weight /= weight_total
    # module_level_weight /= weight_total
    # entity_level_weight /= weight_total

    reward = (
        file_f1_score * file_level_weight
    + module_f1_score * module_level_weight
    + entity_f1_score * entity_level_weight
    )

    return reward, {
        "multilevel_localization_f1_reward": reward,
        "file_reward": file_f1_score,
        "module_reward": module_f1_score,
        "entity_reward": entity_f1_score,
        # "prediction": {
        #     "files": list(predicted_files),
        #     "modules": list(predicted_modules),
        #     "entities": list(predicted_entities),
        # },
        # "ground_truth": {
        #     "files": list(gt_files),
        #     "modules": list(gt_modules),
        #     "entities": list(gt_entities),
        # },
    }
