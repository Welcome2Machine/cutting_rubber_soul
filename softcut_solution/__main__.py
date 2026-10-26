"""CLI entry point for the integrated research workspace."""
import argparse
import json
from pathlib import Path
import sys

from . import api


def main(argv=None):
    parser = argparse.ArgumentParser(description="SoftCut Solution — simulation/video integration")
    parser.add_argument("--root", type=Path, default=api.ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    gui = sub.add_parser("gui", help="Open the local desktop workspace; does not start simulation")
    gui.add_argument("--smoke", action="store_true", help="Construct the GUI and exit after a short self-check")
    sub.add_parser("doctor", help="Inspect both environments without starting numerical work")
    sub.add_parser("inventory", help="List local inputs, accepted cases and result bundles")
    sub.add_parser("simulation-ui", help="Open the copied project's FE workbench")
    export = sub.add_parser("export-fe", help="Export existing audited states, without FE loading")
    export.add_argument("case", type=Path)
    export.add_argument("--frames", type=int, default=21)
    video = sub.add_parser("video-check")
    video.add_argument("case_id")
    batch = sub.add_parser("video-batch", help="Preflight by default; --execute explicitly runs tracking")
    batch.add_argument("manifest", type=Path)
    batch.add_argument("--execute", action="store_true")
    inspection = sub.add_parser("inspect")
    inspection.add_argument("kind", choices=["simulation", "video"])
    inspection.add_argument("path", type=Path)
    comparison = sub.add_parser("compare", help="Compare fixed-ID observations with qualified FE saved states")
    comparison.add_argument("--fe", type=Path, required=True)
    comparison.add_argument("--video", type=Path, required=True)
    comparison.add_argument("--options", type=Path, required=True, help="Explicit registration and comparison JSON")
    args = parser.parse_args(argv)
    try:
        if args.command == "gui":
            from .gui import main as gui_main
            if args.smoke:
                # GUI exposes a smoke argument; no subprocess or research work is started.
                gui_main(root=args.root, smoke=True)
            else:
                gui_main(root=args.root)
            return 0
        if args.command == "doctor":
            result = api.doctor(args.root)
        elif args.command == "inventory":
            result = api.inventory(args.root)
        elif args.command == "simulation-ui":
            result = api.start_fe_workbench(args.root)
        elif args.command == "export-fe":
            result = api.export_fe(args.root, args.case, args.frames)
        elif args.command == "video-check":
            result = api.video_preflight(args.root, args.case_id)
        elif args.command == "video-batch":
            result = api.video_batch(args.root, args.manifest, check_only=not args.execute)
        elif args.command == "inspect":
            result = api.inspect(args.kind, args.path)
        else:
            result = api.compare(args.root, args.fe, args.video, api.read_json(args.options))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
