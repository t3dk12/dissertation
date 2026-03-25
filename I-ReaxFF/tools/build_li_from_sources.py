#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(path):
    with Path(path).open() as f:
        return json.load(f)


def repeated_body_value(pool, prefix, body_len):
    for data in pool:
        p = data.get("p", {})
        for key, value in p.items():
            if not key.startswith(prefix + "_"):
                continue
            rhs = key.split("_", 1)[1]
            parts = rhs.split("-")
            if len(parts) != body_len:
                continue
            if len(set(parts)) == 1 and "X" not in parts:
                return value
    return None


def lookup_key(key, source_p, donors):
    if key in source_p:
        return source_p[key], "source_exact"

    if key.endswith("_Li-Li"):
        base = key[:-6]
        alt = base + "_Li"
        if alt in source_p:
            return source_p[alt], "source_single_to_pair"

    if key.endswith("_Li-Li-Li-Li"):
        alt = key.replace("_Li-Li-Li-Li", "_X-Li-Li-X")
        if alt in source_p:
            return source_p[alt], "source_x_torsion"

    for donor in donors:
        p = donor.get("p", {})
        if key in p:
            return p[key], "donor_exact"

    if key.endswith("_Li-Li"):
        base = key[:-6]
        alt = base + "_Li"
        for donor in donors:
            p = donor.get("p", {})
            if alt in p:
                return p[alt], "donor_single_to_pair"

    if key.endswith("_Li-Li-Li-Li"):
        alt = key.replace("_Li-Li-Li-Li", "_X-Li-Li-X")
        for donor in donors:
            p = donor.get("p", {})
            if alt in p:
                return p[alt], "donor_x_torsion"

    # Last non-template attempt: derive body terms from any all-same-element term in source/donors.
    if key.startswith(("V1_", "V2_", "V3_", "cot1_")) and key.endswith("_Li-Li-Li-Li"):
        prefix = key.split("_", 1)[0]
        val = repeated_body_value([{"p": source_p}] + donors, prefix, 4)
        if val is not None:
            return val, "body4_repeated"

    if key.startswith(("val1_", "val2_", "val4_", "val7_", "pen1_", "coa1_")) and key.endswith("_Li-Li-Li"):
        prefix = key.split("_", 1)[0]
        val = repeated_body_value([{"p": source_p}] + donors, prefix, 3)
        if val is not None:
            return val, "body3_repeated"

    return None, "missing"


def build_li_payload(source, schema, donors):
    out = {}
    for top_key in schema:
        if top_key == "p":
            continue
        # Match li_ffield.json structure exactly for non-p fields.
        out[top_key] = schema[top_key]

    source_p = source.get("p", {})
    out_p = {}
    stats = {}
    missing = []

    for key in schema["p"]:
        val, mode = lookup_key(key, source_p, donors)
        stats[mode] = stats.get(mode, 0) + 1
        if val is None:
            missing.append(key)
        else:
            out_p[key] = val

    out["p"] = out_p
    return out, stats, missing


def main():
    parser = argparse.ArgumentParser(
        description="Build Li-only JSON with no value fallback from li_ffield.json",
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--donors", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    source = load_json(args.source)
    schema = load_json(args.schema)
    donors = [load_json(p) for p in args.donors]

    payload, stats, missing = build_li_payload(source, schema, donors)

    with Path(args.report).open("w") as f:
        print("li_build_report", file=f)
        print(f"source={args.source}", file=f)
        for d in args.donors:
            print(f"donor={d}", file=f)
        for mode in sorted(stats):
            print(f"{mode}={stats[mode]}", file=f)
        print(f"missing_count={len(missing)}", file=f)
        for key in missing:
            print(f"missing={key}", file=f)

    if missing:
        raise SystemExit(
            f"Missing {len(missing)} keys. See report: {args.report}",
        )

    with Path(args.output).open("w") as f:
        json.dump(payload, f, sort_keys=True, indent=2)


if __name__ == "__main__":
    main()
