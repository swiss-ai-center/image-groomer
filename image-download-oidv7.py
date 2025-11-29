
"""
Download images from Google's Open Images Dataset (OID v7) for specific
categories using the FiftyOne library, and save them into an output folder
with one sub-folder per category.

Requirements:
- fiftyone (pip install -r requirements.txt)

Examples:
- python image-download-oidv7.py -c zebra elephant -o data_raw \
  --per-class-limit 300
- python image-download-oidv7.py -f classes.txt -o data_raw --split all \
  --per-class-limit 500

Notes:
- This script downloads only the images that contain the requested classes
  (only_matching=True) to minimize data transfer.
- For OIDv7, labels are provided as detections; we filter images that contain
  at least one detection with the requested label.
"""


import argparse
import logging
import os
import shutil
from typing import List, Sequence

try:
    import fiftyone.zoo as foz
    from fiftyone import ViewField as F
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "FiftyOne is required. Please install with: pip install fiftyone"
    ) from e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download images from Open Images v7 for the given categories and "
            "save them to an output directory with one sub-folder per "
            "category."
        )
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "-c",
        "--categories",
        nargs="+",
        help="Space-separated list of category names (e.g. Cat Dog)",
    )
    grp.add_argument(
        "-f",
        "--categories-file",
        help="Path to a file containing one category per line",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data_raw",
        help="Output root folder where category subfolders will be created",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test", "all"],
        default="all",
        help="Dataset split(s) to pull from (default: all)",
    )
    parser.add_argument(
        "--per-class-limit",
        type=int,
        default=500,
        help="Maximum number of images to save per category (best effort)",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=0,
        help="Increase output verbosity (-v, -vv)",
    )
    return parser.parse_args()


def read_categories(args: argparse.Namespace) -> List[str]:
    if args.categories:
        cats = args.categories
    else:
        with open(args.categories_file, "r", encoding="utf-8") as f:
            cats = [
                ln.strip()
                for ln in f
                if ln.strip() and not ln.startswith("#")
            ]
    # Remove duplicates while preserving order
    seen = set()
    out = []
    for c in cats:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def unique_copy(src: str, dst_dir: str) -> str:
    """Copy src into dst_dir with a unique filename, return destination."""
    ensure_dir(dst_dir)
    base = os.path.basename(src)
    name, ext = os.path.splitext(base)
    dst = os.path.join(dst_dir, base)
    i = 1
    while os.path.exists(dst):
        dst = os.path.join(dst_dir, f"{name}-{i}{ext}")
        i += 1
    shutil.copy2(src, dst)
    return dst


def load_splits_arg(split: str) -> Sequence[str]:
    if split == "all":
        return ("train", "validation", "test")
    return (split,)


def download_for_categories(
    categories: List[str],
    output_root: str,
    splits: Sequence[str],
    per_class_limit: int,
    verbosity: int = 0,
) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(format="%(levelname)s - %(message)s", level=level)

    logging.info("Categories: %s", ", ".join(categories))
    logging.info("Splits: %s", ", ".join(splits))
    logging.info("Per-class limit: %d", per_class_limit)
    logging.info("Output: %s", output_root)

    # We will load one dataset per split to avoid potential API differences
    # and then filter per-class; using only_matching=True limits downloads
    # to only images that contain any of the requested classes
    datasets_by_split = {}
    for split in splits:
        ds = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            label_types=["detections"],
            classes=categories,
            only_matching=True
        )
        datasets_by_split[split] = ds
        logging.info("Loaded split '%s' with %d samples", split, len(ds))

    def _get_detections_field(ds) -> str:
        schema = ds.get_field_schema()
        # Prefer common field names
        for preferred in ("detections", "ground_truth", "labels"):
            if preferred in schema:
                return preferred
        # Otherwise, infer by type name
        for fname, ftype in schema.items():
            tname = getattr(getattr(ftype, "__class__", None), "__name__", "")
            if "detections" in tname.lower():
                return fname
        # Fallback
        return "detections"

    # For each category, aggregate across splits and export up to N images
    for cls in categories:
        saved = 0
        out_dir = os.path.join(output_root, cls.replace(" ", "_"))
        ensure_dir(out_dir)
        logging.info("\nExporting class '%s' -> %s", cls, out_dir)

        for split, ds in datasets_by_split.items():
            if saved >= per_class_limit:
                break
            det_field = _get_detections_field(ds)
            view = ds.filter_labels(det_field, F("label") == cls)

            logging.debug(
                "Split '%s' has %d samples for class '%s'",
                split,
                len(view),
                cls,
            )
            for sample in view:
                if saved >= per_class_limit:
                    break

                # Ensure this image contains only the target category
                # and none of the other requested categories
                detections = sample[det_field]
                if detections is None:
                    continue

                labels_in_image = set(
                    det.label for det in detections.detections
                )
                other_categories = set(categories) - {cls}
                # print('<---')
                # print('labels_in_images=',labels_in_image)
                # print('other_categories=', other_categories)
                # print('current_class=', cls)
                # print('file=', sample.filepath)
                # print('--->')

                # Skip if any other requested category is present
                if labels_in_image & other_categories:
                    logging.debug(
                        "Skipping %s: contains multiple categories %s",
                        sample.filepath,
                        labels_in_image & (other_categories | {cls})
                    )
                    continue

                try:
                    dst = unique_copy(sample.filepath, out_dir)
                    saved += 1
                    logging.debug("Saved %s", dst)
                except Exception as e:  # pragma: no cover
                    logging.warning(
                        "Failed to save %s: %s", sample.filepath, e
                    )

        logging.info("Saved %d images for class '%s'", saved, cls)

    logging.info("Done.")


def main():
    args = parse_args()
    categories = read_categories(args)
    splits = load_splits_arg(args.split)
    download_for_categories(
        categories=categories,
        output_root=args.output,
        splits=splits,
        per_class_limit=args.per_class_limit,
        verbosity=args.verbosity,
    )


if __name__ == "__main__":
    main()
