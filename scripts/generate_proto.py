#!/usr/bin/env python3
"""
Generate Python gRPC code from proto files.

Cross-platform (works on Windows) and intentionally simple.
"""

from __future__ import annotations

import pathlib
import sys

from grpc_tools import protoc


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    proto_dir = repo_root / "proto"
    out_dir = proto_dir / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    proto_file = proto_dir / "scraper.proto"
    if not proto_file.exists():
        print(f"ERROR: missing proto file: {proto_file}", file=sys.stderr)
        return 1

    args = [
        "protoc",
        f"-I{proto_dir}",
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        str(proto_file),
    ]

    print("Running:", " ".join(args))
    rc = protoc.main(args)
    if rc != 0:
        print(f"ERROR: protoc failed with exit code {rc}", file=sys.stderr)
        return rc

    # Add package markers for nicer imports (proto and proto/generated as packages)
    (proto_dir / "__init__.py").write_text("# Proto package\n", encoding="utf-8")
    (out_dir / "__init__.py").write_text("# Generated gRPC code package\n", encoding="utf-8")

    # Fix absolute import in generated *_pb2_grpc.py to be package-relative.
    # grpc_tools sometimes generates: `import scraper_pb2 as scraper__pb2`
    # which fails when code lives under proto/generated.
    grpc_file = out_dir / "scraper_pb2_grpc.py"
    if grpc_file.exists():
        text = grpc_file.read_text(encoding="utf-8")
        text2 = text.replace(
            "import scraper_pb2 as scraper__pb2",
            "from . import scraper_pb2 as scraper__pb2",
        )
        if text2 != text:
            grpc_file.write_text(text2, encoding="utf-8")
            print("Fixed import in:", grpc_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


