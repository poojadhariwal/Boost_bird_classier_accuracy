from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

from birdsound.data import build_metadata_frame, scan_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build metadata CSV for the bird audio dataset.")
    parser.add_argument("--data-dir", required=True, help="Path to dataset root folder.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    args = parser.parse_args()

    records = scan_dataset(args.data_dir)
    frame = build_metadata_frame(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Saved metadata to {output}")


if __name__ == "__main__":
    main()
