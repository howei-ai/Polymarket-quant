#!/usr/bin/env python3
"""Replay frozen pretrade feature rows through the K3/Jev agent shadow (SHADOW only).

  --plan-only   verify the pinned inputs and print row, drop and evidence counts; no model calls.
  otherwise     attempt up to --max-rows rows within --budget-usd and append SHADOW artifacts
                to --out-dir (outside this repository, the inputs and data/), resuming past rows
                that already have a final attempt.

Rows are redacted before they reach a model (hashed condition id, timestamps shifted to a
synthetic epoch, absolute BTC price levels dropped); the real ids stay in a local mapping file.
Credentials come from the chosen route (official: OPENAI_API_KEY; relay_mtu:
STD0_RELAY_API_KEY). A relay run needs --allow-unverified-relay and
its artifacts are labelled UNVERIFIED_RELAY. Nothing here writes data/, the formal cohort or
frozen artifacts, submits orders, publishes, or runs a backtest.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import json
import math
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from std0_quant.config import load_settings, resolve_path  # noqa: E402
from std0_quant.research.agent_shadow import provider_clients  # noqa: E402
from std0_quant.research.agent_shadow.model_registry import ROUTES, CostBudget, binding_for  # noqa: E402
from std0_quant.research.agent_shadow.replay import (  # noqa: E402
    REPLAY_VERSION, ReplayWriter, build_replay_plan, load_frozen_inputs, replay_run_binding, run_replay,
)
from std0_quant.research.agent_shadow.runner import ShadowRunner  # noqa: E402


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


PROXY_SCHEMES = ("http", "https", "socks5", "socks5h")


def _proxy_label(proxy_url: str | None) -> str | None:
    """Scheme, host and port only: credentials in a proxy URL are never written to disk."""
    if proxy_url is None:
        return None
    parts = urlsplit(proxy_url)
    if parts.scheme not in PROXY_SCHEMES or not parts.hostname:
        raise SystemExit(f"--proxy-url must be {'/'.join(PROXY_SCHEMES)}://host[:port]")
    return f"{parts.scheme}://{parts.hostname}" + (f":{parts.port}" if parts.port else "")


def _forbidden_roots(input_dir: Path) -> tuple[Path, ...]:
    """The repository, the inputs and every configured data path (any may be absolute)."""
    roots = [ROOT, input_dir]
    try:
        settings = load_settings(project_root=ROOT)
        roots.extend(resolve_path(settings, name) for name in type(settings.paths).model_fields)
    except Exception as exc:  # settings unreadable: refuse to guess where data/ is
        raise SystemExit(f"cannot resolve the configured data paths: {type(exc).__name__}") from None
    return tuple(roots)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="directory holding the pinned frozen feature and provenance parquet files")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--route", choices=sorted(ROUTES), default="official")
    parser.add_argument("--allow-unverified-relay", action="store_true")
    parser.add_argument("--proxy-url")
    parser.add_argument("--include-ineligible", action="store_true",
                        help="also replay rows that fail coverage (they are BLOCKED before any call)")
    return parser


def _check_run_arguments(args: argparse.Namespace) -> None:
    if args.out_dir is None or args.max_rows is None or args.budget_usd is None:
        raise SystemExit("--out-dir, --max-rows and --budget-usd are required unless --plan-only")
    if args.max_rows < 1:
        raise SystemExit("--max-rows must be at least 1")
    if not math.isfinite(args.budget_usd) or args.budget_usd <= 0:
        raise SystemExit("--budget-usd must be a finite positive amount")
    if ROUTES[args.route].provenance != "OFFICIAL" and not args.allow_unverified_relay:
        raise SystemExit(f"route {args.route} needs --allow-unverified-relay")
    _proxy_label(args.proxy_url)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.plan_only:
        _check_run_arguments(args)
    inputs = load_frozen_inputs(args.input_dir)
    plan = build_replay_plan(inputs["features"]["records"], inputs["provenance"]["records"],
                             eligible_only=not args.include_ineligible)
    rows = plan.rows
    summary_plan = {
        "replay_version": REPLAY_VERSION, "rows": len(rows), "dropped": dict(plan.dropped),
        "evidence_all_true": sum(all((r.evidence.coverage_valid, r.evidence.provenance_valid,
                                      r.evidence.sanity_valid)) for r in rows),
        "evidence_failures": dict(Counter(failure for r in rows for failure in r.evidence_failures)),
        "redaction": "condition_id hashed; timestamps shifted to a synthetic epoch; absolute BTC prices dropped",
    }
    if args.plan_only:
        print(json.dumps(summary_plan, indent=2, sort_keys=True))
        return 0
    writer = ReplayWriter(args.out_dir, forbidden_roots=_forbidden_roots(args.input_dir))
    writer.attempt_state(model_ids=(binding_for("K3").model_id, binding_for("JEV").model_id))
    writer.validate_run_binding(replay_run_binding(args.route))
    budget = CostBudget(args.budget_usd)
    with (closing(provider_clients.build_openai_client(proxy_url=args.proxy_url, route=args.route)) as k3_sdk,
          closing(provider_clients.build_openai_client(proxy_url=args.proxy_url, route=args.route)) as jev_sdk):
        k3, jev = provider_clients.build_shadow_models(
            k3_sdk, jev_sdk, budget=budget, network_enabled=True, route=args.route)
        runner = ShadowRunner(k3, jev, allow_unverified_relay=args.allow_unverified_relay)
        manifest = {
            **summary_plan, "code_head": _git_head(), "route": args.route, "provenance": ROUTES[args.route].provenance,
            "models": {role: {"provider": binding_for(role).provider, "model_id": binding_for(role).model_id,
                              "effort": binding_for(role).effort} for role in ("K3", "JEV")},
            "max_rows": args.max_rows, "budget_usd": args.budget_usd, "proxy": _proxy_label(args.proxy_url),
            "inputs": {name: {"file": item["path"].name, "sha256": item["sha256"], "rows": len(item["records"])}
                       for name, item in inputs.items()},
            "execution_allowed": False, "formal_cohort_effect": False,
        }
        writer.write_once("run_manifest", manifest)
        summary = run_replay(rows, runner, writer, budget, max_rows=args.max_rows)
        writer.write_once("run_summary", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
