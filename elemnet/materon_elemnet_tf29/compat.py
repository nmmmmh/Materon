from __future__ import absolute_import, division, print_function

import os
import platform
from typing import Any, Dict, Optional


def package_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version
    except ImportError:
        from importlib_metadata import version  # type: ignore
    try:
        return str(version(name))
    except Exception:
        return None


def collect_environment() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow": package_version("tensorflow"),
        "kgcnn": package_version("kgcnn"),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "scikit_learn": package_version("scikit-learn"),
        "pymatgen": package_version("pymatgen"),
        "matminer": package_version("matminer"),
        "matbench": package_version("matbench"),
        "pyyaml": package_version("PyYAML"),
        "joblib": package_version("joblib"),
    }


def parse_version(value: Optional[str]):
    if value is None:
        return tuple()
    parts = []
    for token in value.split("."):
        digits = ""
        for char in token:
            if char.isdigit():
                digits += char
            else:
                break
        if digits:
            parts.append(int(digits))
        else:
            break
    return tuple(parts)


def validate_target_environment(strict: bool = True) -> Dict[str, Any]:
    env = collect_environment()
    checks = {
        "python_3_9": parse_version(env["python"])[:2] == (3, 9),
        "tensorflow_2_9": parse_version(env["tensorflow"])[:2] == (2, 9),
        "kgcnn_2_1": parse_version(env["kgcnn"])[:2] == (2, 1),
        "matbench_0_6": parse_version(env["matbench"])[:2] == (0, 6),
        "pymatgen_2022_9": parse_version(env["pymatgen"])[:2] == (2022, 9),
    }
    result = {"environment": env, "checks": checks, "all_pass": all(checks.values())}
    if strict and not result["all_pass"]:
        failed = [name for name, status in checks.items() if not status]
        raise RuntimeError("Target environment check failed: %s; detected=%r" % (failed, env))
    return result


def configure_tensorflow(seed: int, deterministic: bool = True):
    """Import/configure TensorFlow 2.9 once per resident worker.

    Deterministic kernels are retained to reduce seed/fold drift.
    There is no duplicate replay and training remains on the fast
    model.fit/tf.data path.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    if deterministic:
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    import tensorflow as tf
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
    except Exception:
        pass
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    tf.random.set_seed(int(seed))
    return tf
