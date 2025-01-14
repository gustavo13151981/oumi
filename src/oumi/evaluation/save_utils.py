from pathlib import Path
from typing import Any, Optional

from oumi.core.configs import (
    EvaluationTaskParams,
    GenerationParams,
    InferenceConfig,
    ModelParams,
)
from oumi.core.configs.params.evaluation_params import EvaluationPlatform
from oumi.utils.git_utils import get_git_revision_hash, get_git_tag
from oumi.utils.serialization_utils import json_serializer
from oumi.utils.version_utils import get_python_package_versions

# Output filenames for saving evaluation results.
OUTPUT_FILENAME_PLATFORM_RESULTS = "platform_results.json"
OUTPUT_FILENAME_PLATFORM_TASK_CONFIG = "platform_task_config.json"
OUTPUT_FILENAME_TASK_PARAMS = "task_params.json"
OUTPUT_FILENAME_MODEL_PARAMS = "model_params.json"
OUTPUT_FILENAME_GENERATION_PARAMS = "generation_params.json"
OUTPUT_FILENAME_INFERENCE_CONFIG = "inference_config.json"
OUTPUT_FILENAME_PACKAGE_VERSIONS = "package_versions.json"
OUTPUT_FILENAME_OUMI_REPO_VERSION = "oumi_repo_version.json"


def _save_to_file(output_path: Path, data: Any) -> None:
    """Serialize and save `data` to `output_path`."""
    with open(output_path, "w") as file_out:
        file_out.write(json_serializer(data))


def save_evaluation_output(
    base_output_dir: str,
    platform: EvaluationPlatform,
    platform_results: dict[str, Any],
    platform_task_config: dict[str, Any],
    task_params: EvaluationTaskParams,
    start_time_str: str,
    elapsed_time_sec: float,
    model_params: ModelParams,
    generation_params: GenerationParams,
    inference_config: Optional[InferenceConfig] = None,
) -> None:
    """Writes configuration settings and evaluations outputs to files.

    Args:
        base_output_dir: The directory where the evaluation results will be saved.
            A subdirectory with the name `<base_output_dir> / <platform>_<time>`
            will be created to retain all files related to this evaluation.
        platform: The evaluation platform used (e.g., "lm_harness", "alpaca_eval").
        platform_results: The evaluation results (metrics and their values) to save.
        platform_task_config: The platform-specific task configuration to save.
        task_params: The Oumi task parameters that were used for this evaluation.
        start_time_str: A string containing the start date/time of this evaluation.
        elapsed_time_sec: The duration of the evaluation (in seconds).
        model_params: The model parameters that were used in the evaluation.
        generation_params: The generation parameters that were used in the evaluation.
        inference_config: The inference configuration used in the evaluation
        (if inference is required for the corresponding evaluation platform).
    """
    # Create the output directory: `<base_output_dir> / <platform>_<time>`.
    output_dir = Path(base_output_dir) / f"{platform.value}_{start_time_str}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all evaluation metrics, start date/time, and duration.
    platform_results["duration_sec"] = elapsed_time_sec
    platform_results["start_time"] = start_time_str
    _save_to_file(output_dir / OUTPUT_FILENAME_PLATFORM_RESULTS, platform_results)

    # Save platform-specific task configuration.
    _save_to_file(
        output_dir / OUTPUT_FILENAME_PLATFORM_TASK_CONFIG, platform_task_config
    )

    # Save Oumi's task parameters/configuration.
    _save_to_file(output_dir / OUTPUT_FILENAME_TASK_PARAMS, task_params)

    # Save all relevant Oumi configurations.
    _save_to_file(output_dir / OUTPUT_FILENAME_MODEL_PARAMS, model_params)
    _save_to_file(output_dir / OUTPUT_FILENAME_GENERATION_PARAMS, generation_params)
    if inference_config:
        _save_to_file(output_dir / OUTPUT_FILENAME_INFERENCE_CONFIG, inference_config)

    # Save python environment (package versions).
    package_versions = get_python_package_versions()
    _save_to_file(output_dir / OUTPUT_FILENAME_PACKAGE_VERSIONS, package_versions)

    # Save the Oumi repository's versioning info.
    repo_version_dict = {
        "oumi_git_commit_hash": get_git_revision_hash(),
        "oumi_git_tag": get_git_tag(),
    }
    _save_to_file(output_dir / OUTPUT_FILENAME_OUMI_REPO_VERSION, repo_version_dict)
