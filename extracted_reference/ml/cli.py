"""Command-line interface for the SIH26170 screening pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from screening_engine import (
    build_feature_frame,
    evaluate,
    generate_demo_data,
    load_artifact,
    load_input,
    make_html_report,
    save_artifact,
    score,
    train,
    validate_frame,
)


def _validated(path: str):
    validation = validate_frame(load_input(path))
    if validation.clean.empty:
        raise ValueError("No valid measurements remain after validation. Inspect the quarantine output.")
    return validation


def _write_csv(frame, destination: str | None) -> None:
    if destination:
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)


def command_validate(args: argparse.Namespace) -> None:
    result = _validated(args.input)
    _write_csv(result.clean, args.clean_output)
    _write_csv(result.quarantine, args.quarantine_output)
    print(json.dumps(result.summary, indent=2, sort_keys=True))


def command_generate_demo(args: argparse.Namespace) -> None:
    frame = generate_demo_data(seed=args.seed, lots=args.lots, components_per_lot=args.components_per_lot)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(json.dumps({"output": args.output, "rows": len(frame), "label": "SYNTHETIC_DEMONSTRATION_ONLY"}, indent=2))


def command_train(args: argparse.Namespace) -> None:
    validation = _validated(args.input)
    overrides = json.loads(Path(args.policy_json).read_text(encoding="utf-8")) if args.policy_json else None
    artifact = train(build_feature_frame(validation.clean), seed=args.seed, policy_overrides=overrides)
    save_artifact(artifact, args.model_output)
    print(json.dumps({"model_output": args.model_output, "training_rows": artifact["training_rows"], "candidate_validation": artifact["candidate_validation"]}, indent=2))


def command_evaluate(args: argparse.Namespace) -> None:
    validation = _validated(args.input)
    artifact = load_artifact(args.model)
    metrics = evaluate(build_feature_frame(validation.clean), artifact)
    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def command_score(args: argparse.Namespace) -> None:
    validation = _validated(args.input)
    results = score(build_feature_frame(validation.clean), load_artifact(args.model))
    _write_csv(results, args.output)
    print(json.dumps({"output": args.output, "screened_series": len(results), "actions": results["action"].value_counts().to_dict()}, indent=2))


def command_report(args: argparse.Namespace) -> None:
    validation = _validated(args.input)
    artifact = load_artifact(args.model)
    features = build_feature_frame(validation.clean)
    results = score(features, artifact)
    metrics = evaluate(features, artifact) if features["has_target_168h"].any() else None
    Path(args.output).write_text(make_html_report(results, metrics), encoding="utf-8")
    print(json.dumps({"output": args.output, "screened_series": len(results)}, indent=2))


def parser() -> argparse.ArgumentParser:
    main = argparse.ArgumentParser(description="SIH26170 deterministic burn-in screening pipeline")
    sub = main.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-data", help="Validate a CSV, XLSX, XLS, or JSON file.")
    validate.add_argument("--input", required=True)
    validate.add_argument("--clean-output")
    validate.add_argument("--quarantine-output")
    validate.set_defaults(func=command_validate)
    demo = sub.add_parser("generate-demo-data", help="Generate clearly labelled synthetic demonstration data.")
    demo.add_argument("--output", required=True)
    demo.add_argument("--seed", type=int, default=26170)
    demo.add_argument("--lots", type=int, default=5)
    demo.add_argument("--components-per-lot", type=int, default=28)
    demo.set_defaults(func=command_generate_demo)
    training = sub.add_parser("train", help="Train a deterministic forecasting artifact.")
    training.add_argument("--input", required=True)
    training.add_argument("--model-output", required=True)
    training.add_argument("--seed", type=int, default=26170)
    training.add_argument("--policy-json", help="Optional JSON object overriding documented safety-policy settings.")
    training.set_defaults(func=command_train)
    evaluation = sub.add_parser("evaluate", help="Evaluate a model artifact on a validated dataset.")
    evaluation.add_argument("--input", required=True)
    evaluation.add_argument("--model", required=True)
    evaluation.add_argument("--output")
    evaluation.set_defaults(func=command_evaluate)
    scoring = sub.add_parser("predict", help="Batch-score early burn-in measurements.")
    scoring.add_argument("--input", required=True)
    scoring.add_argument("--model", required=True)
    scoring.add_argument("--output", required=True)
    scoring.set_defaults(func=command_score)
    report = sub.add_parser("report", help="Generate an auditable HTML screening report.")
    report.add_argument("--input", required=True)
    report.add_argument("--model", required=True)
    report.add_argument("--output", required=True)
    report.set_defaults(func=command_report)
    return main


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
