#!/usr/bin/env python
"""Phase-0 dataset audit.

Walks the LRS3 layout, verifies every utterance's file triplet / pair, loads
every ``.tokens.pt`` to record exact frame counts (=> durations), parses
transcripts, normalizes them with the frozen canonical function, and writes:

- ``data_index.parquet``  : one row per utterance (the project's single source of truth)
- ``audit_report.json``   : machine-readable counts/anomalies
- stdout summary          : human-readable report

Run from the project root (pymax venv):
    python scripts/audit_data.py [--root datasets/LRS3] [--out-dir .] [--sample N]

Verified Phase-0 facts this script re-checks: tokens are [8, T] int64 over
[0, 1024); T == ceil(n_samples/320); trainval is per-video folders with numeric
stems that RESTART per folder (hence "<video>/<stem>" utterance ids); official
test is flat wav+tokens pairs with no transcripts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import text_norm  # noqa: E402
from config import ProtocolConfig, load_protocol, universe_budget  # noqa: E402

MAX_EXAMPLES_KEPT = 50  # anomaly lists are truncated in the JSON report


class Anomalies:
    def __init__(self) -> None:
        self.counters: Counter[str] = Counter()
        self.examples: dict[str, list[str]] = {}

    def add(self, kind: str, detail: str) -> None:
        self.counters[kind] += 1
        self.examples.setdefault(kind, [])
        if len(self.examples[kind]) < MAX_EXAMPLES_KEPT:
            self.examples[kind].append(detail)

    def to_json(self) -> dict:
        return {
            "counts": dict(self.counters),
            "examples": self.examples,
            "examples_truncated_after": MAX_EXAMPLES_KEPT,
        }


def _sorted_numeric(items: list[str]) -> list[str]:
    """Sort stems numerically when possible, lexically otherwise."""
    return sorted(items, key=lambda s: (0, int(s)) if s.isdigit() else (1, s))


def scan_trainval(root: Path, proto: ProtocolConfig, anomalies: Anomalies,
                  limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    video_dirs = sorted(d for d in root.joinpath("trainval").iterdir() if d.is_dir())
    n_videos_seen = 0
    for vdir in tqdm(video_dirs, desc="trainval videos"):
        if limit is not None and len(rows) >= limit:
            break
        n_videos_seen += 1
        txts = _sorted_numeric([p.name for p in vdir.glob("*.txt")])
        for txt_name in txts:
            stem = txt_name[:-4]
            uid = f"{vdir.name}/{stem}"
            mp4 = vdir / f"{stem}.mp4"
            tok = vdir / f"{stem}.tokens.pt"

            if not mp4.exists():
                anomalies.add("trainval_missing_mp4", uid)
            if not tok.exists():
                anomalies.add("trainval_missing_tokens", uid)
                continue

            raw_txt = (vdir / txt_name).read_text(encoding="utf-8")
            if not raw_txt.strip():
                anomalies.add("trainval_empty_transcript", uid)
                conf, text_raw = None, ""
            else:
                try:
                    t = text_norm.parse_lrs3_transcript(raw_txt)
                    conf, text_raw = t.conf, t.text_raw
                    if not t.text_raw.strip():
                        anomalies.add("trainval_blank_text_field", uid)
                except ValueError:
                    anomalies.add("trainval_unparsable_transcript", uid)
                    conf, text_raw = None, ""

            row = _token_row(
                uid, split="trainval", video_id=vdir.name, stem=stem,
                tokens_path=str(tok), audio_path=str(mp4), audio_kind="mp4",
                txt_path=str(vdir / txt_name), conf=conf, text_raw=text_raw,
                proto=proto, anomalies=anomalies,
            )
            rows.append(row)

            # orphan mp4 (mp4 without any txt) is caught on the mp4 side below
        for p in vdir.glob("*.mp4"):
            if not (vdir / (p.stem + ".txt")).exists():
                anomalies.add("trainval_orphan_mp4_no_txt", f"{vdir.name}/{p.name}")
    return rows


def scan_test(root: Path, proto: ProtocolConfig, anomalies: Anomalies,
              limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    tdir = root / "test"
    tok_files = _sorted_numeric([p.name for p in tdir.glob("*.tokens.pt")])
    for name in tqdm(tok_files, desc="test clips"):
        if limit is not None and len(rows) >= limit:
            break
        stem = name[: -len(".tokens.pt")]
        wav = tdir / f"{stem}.wav"
        if not wav.exists():
            anomalies.add("test_missing_wav", stem)
            continue
        # recovered transcripts may exist post-recovery; absent ones are NOT anomalies
        txt = tdir / f"{stem}.txt"
        conf, text_raw = None, ""
        if txt.exists():
            try:
                t = text_norm.parse_lrs3_transcript(txt.read_text(encoding="utf-8"))
                conf, text_raw = t.conf, t.text_raw
            except ValueError:
                anomalies.add("test_unparsable_transcript", stem)

        rows.append(_token_row(
            uid=stem, split="test", video_id="", stem=stem,
            tokens_path=str(txt.parent / name), audio_path=str(wav), audio_kind="wav",
            txt_path=str(txt) if txt.exists() else None,
            conf=conf, text_raw=text_raw, proto=proto, anomalies=anomalies,
        ))
    return rows


def _token_row(uid, *, split, video_id, stem, tokens_path, audio_path, audio_kind,
               txt_path, conf, text_raw, proto, anomalies) -> dict:
    try:
        tokens = torch.load(tokens_path, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001 - any unreadable tensor is an anomaly row
        anomalies.add(f"{split}_unreadable_tokens", f"{uid}: {e}")
        return {
            "utterance_id": uid, "split": split, "video_id": video_id, "stem": stem,
            "tokens_path": tokens_path, "audio_path": audio_path, "audio_kind": audio_kind,
            "txt_path": txt_path, "n_tokens": 0, "duration_s": 0.0,
            "selectable": False,  # unreadable tokens can never be selected
            "conf": conf, "text_raw": text_raw, "text_norm": "", "n_chars_norm": 0,
        }

    shape_ok = (
        isinstance(tokens, torch.Tensor) and tokens.ndim == 2
        and tokens.shape[0] == proto.n_rvq_streams
    )
    if not shape_ok:
        got = tuple(tokens.shape) if isinstance(tokens, torch.Tensor) else type(tokens).__name__
        anomalies.add(f"{split}_bad_token_shape", f"{uid}: {got}")
    elif tokens.min().item() < 0 or tokens.max().item() >= proto.codebook_size:
        anomalies.add(
            f"{split}_token_range_violation",
            f"{uid}: [{tokens.min().item()}, {tokens.max().item()}]",
        )

    n_tokens = int(tokens.shape[-1]) if isinstance(tokens, torch.Tensor) and tokens.ndim == 2 else 0
    if 0 < n_tokens < proto.min_crop_frames:
        anomalies.add(f"{split}_too_short_for_crops", f"{uid}: T={n_tokens}")

    # Selectable universe (locked rule): trainval utts with >= min_crop_frames.
    # Everything outside it is invisible to manifests, selections, and budgets.
    selectable = split == "trainval" and n_tokens >= proto.min_crop_frames

    norm = text_norm.normalize_text(text_raw) if text_raw else ""
    return {
        "utterance_id": uid, "split": split, "video_id": video_id, "stem": stem,
        "tokens_path": tokens_path, "audio_path": audio_path, "audio_kind": audio_kind,
        "txt_path": txt_path, "n_tokens": n_tokens,
        "duration_s": round(n_tokens / proto.token_hz, 4),
        "selectable": selectable,
        "conf": conf, "text_raw": text_raw, "text_norm": norm,
        "n_chars_norm": len(norm),
    }


def summarize(rows: list[dict], proto: ProtocolConfig) -> dict:
    df = pd.DataFrame(rows)
    report: dict = {"protocol": {"sample_rate": proto.sample_rate, "token_hz": proto.token_hz,
                                 "codebook_size": proto.codebook_size, "n_rvq_streams": proto.n_rvq_streams}}
    for split, g in df.groupby("split"):
        report[split] = {
            "utterances": int(len(g)),
            "videos": int(g["video_id"].replace("", pd.NA).dropna().nunique()),
            "hours": round(float(g["duration_s"].sum()) / 3600.0, 3),
            "duration_s_mean": round(float(g["duration_s"].mean()), 3),
            "duration_s_p95": round(float(g["duration_s"].quantile(0.95)), 3),
            "with_transcripts": int((g["text_norm"] != "").sum()),
            "empty_norm_transcripts": int(((g["text_norm"] == "") & g["txt_path"].notna()).sum()),
            "conf_histogram": {
                str(int(k)): int(v) for k, v in
                sorted(Counter(int(c) for c in g["conf"].dropna()).items())
            },
            "budget_at_25pct_hours": round(float(g["duration_s"].sum()) * proto.budget_fraction / 3600.0, 3),
        }
    report["_total_hours"] = round(float(df["duration_s"].sum()) / 3600.0, 3)

    # The selectable universe: the ONLY pool manifests/selections/budgets may draw from.
    uni = df[df["selectable"] == True]  # noqa: E712 - pandas boolean mask
    report["universe"] = {
        "definition": f"trainval utts with n_tokens >= {proto.min_crop_frames}",
        "utterances": int(len(uni)),
        "hours": round(float(uni["duration_s"].sum()) / 3600.0, 3),
        "budget_utts": universe_budget(int(len(uni)), proto.budget_fraction),
        "budget_rule": "round(budget_fraction * |universe|), BY UTTERANCE COUNT",
        "excluded_short": int(((df["split"] == "trainval") & (df["selectable"] == False)).sum()),  # noqa: E712
    }
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("datasets/LRS3"))
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    ap.add_argument("--sample", type=int, default=None,
                    help="quick mode: only the first N folders/files per split")
    ap.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs" / "protocol.yaml")
    args = ap.parse_args(argv)

    proto = load_protocol(args.protocol)
    anomalies = Anomalies()

    rows = scan_trainval(args.root, proto, anomalies, args.sample)
    rows += scan_test(args.root, proto, anomalies, args.sample)
    df = pd.DataFrame(rows)
    df = df.sort_values(["split", "utterance_id"]).reset_index(drop=True)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "data_index.parquet"
    df.to_parquet(index_path, index=False)

    report = summarize(rows, proto)
    report["anomalies"] = anomalies.to_json()
    report["index_path"] = str(index_path)
    report_path = out_dir / "audit_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== AUDIT SUMMARY ===")
    for split in ("trainval", "test"):
        if split in report:
            r = report[split]
            print(f"[{split}] utts={r['utterances']}  videos={r['videos']}  "
                  f"hours={r['hours']}  25%={r['budget_at_25pct_hours']}h  "
                  f"transcripts={r['with_transcripts']}")
            print(f"        mean={r['duration_s_mean']}s  p95={r['duration_s_p95']}s  "
                  f"conf_hist={r['conf_histogram']}")
    print(f"TOTAL hours: {report['_total_hours']}")
    u = report["universe"]
    print(f"UNIVERSE: {u['utterances']} utts ({u['hours']}h) [{u['definition']}]  "
          f"budget@{round(proto.budget_fraction * 100)}%={u['budget_utts']}utts  "
          f"excluded_short={u['excluded_short']}")
    print(f"Anomaly counts: {dict(anomalies.counters) or 'NONE'}")
    print(f"Wrote {index_path} ({len(df)} rows) and {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
