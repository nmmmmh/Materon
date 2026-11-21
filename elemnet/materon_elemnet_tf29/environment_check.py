from __future__ import absolute_import, division, print_function

import argparse
import json

from .compat import validate_target_environment


def main():
    parser = argparse.ArgumentParser(description="Check the exact kgcnn_old target environment.")
    parser.add_argument("--allow-version-drift", action="store_true")
    args = parser.parse_args()
    result = validate_target_environment(strict=not args.allow_version_drift)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["all_pass"]:
        print("ENVIRONMENT_CHECK_PASS")


if __name__ == "__main__":
    main()
