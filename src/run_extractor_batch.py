#!/usr/bin/env python3
import argparse, os, sys, subprocess, pathlib

def main():
    ap = argparse.ArgumentParser(description="Batch run extractor.py over a directory of PDFs")
    ap.add_argument("--in-dir", required=True, help="Directory containing PDFs (recursive)")
    ap.add_argument("--outdir", required=True, help="Directory to write *.archive.json and *.llm.json")
    ap.add_argument("--workers", type=int, default=1, help="(future) parallelism")
    args = ap.parse_args()

    in_dir = pathlib.Path(args.in_dir)
    out_dir = pathlib.Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = [str(p) for p in in_dir.rglob("*.pdf")]
    if not pdfs:
        print(f"[batch-extractor] No PDFs found under {in_dir}", file=sys.stderr)
        sys.exit(2)

    ok, fail = 0, 0
    for pdf in sorted(pdfs):
        cmd = [sys.executable, "/app/extractor.py", pdf, "--outdir", str(out_dir)]
        print("[extract]", " ".join(cmd), flush=True)
        try:
            subprocess.run(cmd, check=True)
            ok += 1
        except subprocess.CalledProcessError as e:
            print(f"[extract][ERROR] {pdf}: {e}", file=sys.stderr)
            fail += 1

    print(f"[batch-extractor] Done: {ok} ok, {fail} failed; outputs at {out_dir}")
    sys.exit(0 if ok > 0 and fail == 0 else (0 if ok > 0 else 1))

if __name__ == "__main__":
    main()
