#!/usr/bin/env python3
"""Dump a ReaxFF force field into the JSON schema consumed by json_to_ffield.py."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from irff.reaxfflib import read_ffield


def _normalize_value(value: Any) -> Any:
    """Convert numpy scalars to built-in Python types for JSON serialization."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def ffield_to_json(libfile: str, outfile: str) -> None:
    """Read a ReaxFF ffield and emit a JSON file with the same keys json_to_ffield expects."""

    meta = _try_load_meta(libfile)
    if meta is not None:
        payload = meta
    else:
        p, zpe, *_ = read_ffield(libfile=libfile, zpe=True)
        if p is None:
            raise FileNotFoundError(f"Could not read force field from {libfile}")

        clean_p = {key: _normalize_value(value) for key, value in p.items()}
        clean_zpe = {key: _normalize_value(value) for key, value in zpe.items()}

        payload = {
            "p": clean_p,
            "m": None,
            "mf_layer": None,
            "be_layer": None,
            "vdw_layer": None,
            "EnergyFunction": 0,
            "VdwFunction": 0,
            "MessageFunction": 0,
            "messages": 1,
            "MolEnergy": {},
            "zpe": clean_zpe,
            "rcut": None,
            "rEquilibrium": None,
            "rcutBond": None,
        }

    with open(outfile, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a ReaxFF ffield file into the JSON format consumed by json_to_ffield.py",
    )
    parser.add_argument("-f", "--ffield", default="ffield", help="input force field file")
    parser.add_argument(
        "-o",
        "--output",
        default="ffield.json",
        help="target JSON file",
    )
    args = parser.parse_args()
    ffield_to_json(args.ffield, args.output)


def _try_load_meta(libfile: str) -> dict | None:
    meta_path = f"{libfile}.meta.json"
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as mf:
            return json.load(mf)
    return None


if __name__ == "__main__":
    main()
