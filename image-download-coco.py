"""
Download images from the COCO dataset (2014/2017) for specific categories
using the FiftyOne library, and save them into an output folder with one
sub-folder per category.

Requirements:
- fiftyone (pip install -r requirements.txt)

Examples:
- python image-download-coco.py -c cat dog -o data_raw --per-class-limit 300
- python image-download-coco.py -f classes.txt -o data_raw --split all \
  --per-class-limit 500

Notes:
- By default, this script ensures that images saved for a class do NOT also
  contain any of the other requested classes (exclusive per-requested-class).
- To perform this exclusivity check correctly, we load all labels for matching
  images (only_matching=False) so we can see other objects in the scene.
- If you want to be even stricter (keep images that contain only the target
  class and nothing else at all), use --exclude-any-non-target.
- COCO category names are lowercase (e.g. "cat", "dog", "zebra"). The script
  will match labels case-insensitively.
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
            "Download images from COCO for the given categories and save them "
            "to an output directory with one sub-folder per category."
        )
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "-c",
        "--categories",
        nargs="+",
        help="Space-separated list of category names (e.g. cat dog)",
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
        "--dataset",
        choices=["coco-2017", "coco-2014"],
        default="coco-2017",
        help="COCO dataset version to use (default: coco-2017)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test", "all"],
        default="train",
        help=(
            "Dataset split(s) to pull from. Note: COCO 'test' has no labels "
            "for filtering."
        ),
    )
    parser.add_argument(
        "--per-class-limit",
        type=int,
        default=500,
        help="Maximum number of images to save per category (best effort)",
    )
    parser.add_argument(
        "--exclude-any-non-target",
        action="store_true",
        help=(
            "If set, keep only images whose labels are exclusively the "
            "current target class (i.e., no other objects at all)."
        ),
    )
    parser.add_argument(
        "--case-insensitive",
        action="store_true",
        default=True,
        help="Match categories case-insensitively (default: on)",
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
        return ("train", "validation")
    return (split,)


def download_for_categories(
    categories: List[str],
    output_root: str,
    dataset_name: str,
    splits: Sequence[str],
    per_class_limit: int,
    exclude_any_non_target: bool,
    case_insensitive: bool,
    verbosity: int = 0,
) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(format="%(levelname)s - %(message)s", level=level)

    # Normalize for matching
    if case_insensitive:
        categories_match = [c.lower() for c in categories]
    else:
        categories_match = categories[:]

    logging.info("Categories: %s", ", ".join(categories))
    logging.info("Splits: %s", ", ".join(splits))
    logging.info("Per-class limit: %d", per_class_limit)
    logging.info("Output: %s", output_root)
    logging.info("Dataset: %s", dataset_name)

    # To check for other objects in images, we must load all labels for
    # matching images. Hence only_matching=False here.
    only_matching = False

    if "test" in splits:
        logging.warning(
            "COCO 'test' split contains no labels; filtering will yield 0"
        )

    datasets_by_split = {}
    for split in splits:
        ds = foz.load_zoo_dataset(
            dataset_name,
            split=split,
            label_types=["detections"],
            classes=categories_match,
            only_matching=only_matching,
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
    for idx, cls in enumerate(categories):
        cls_match = categories_match[idx]
        saved = 0
        out_dir = os.path.join(output_root, cls.replace(" ", "_"))
        ensure_dir(out_dir)
        logging.info("\nExporting class '%s' -> %s", cls, out_dir)

        for split, ds in datasets_by_split.items():
            if saved >= per_class_limit:
                break
            det_field = _get_detections_field(ds)

            # First, keep samples that contain the target class
            # Note: compare on normalized labels if case-insensitive
            if case_insensitive:
                view = ds.filter_labels(
                    det_field,
                    F("label").lower() == cls_match,
                )
            else:
                view = ds.filter_labels(det_field, F("label") == cls_match)

            logging.debug(
                "Split '%s' has %d samples for class '%s'",
                split,
                len(view),
                cls,
            )

            for sample in view:
                if saved >= per_class_limit:
                    break

                detections = sample[det_field]
                if detections is None:
                    continue

                labels_in_image = set(
                    (det.label.lower() if case_insensitive else det.label)
                    for det in detections.detections
                )

                # If requested, exclude images that contain ANY non-target
                # labels
                if exclude_any_non_target:
                    if any(lbl != cls_match for lbl in labels_in_image):
                        logging.debug(
                            "Skipping %s: contains non-target labels %s",
                            sample.filepath,
                            sorted(labels_in_image),
                        )
                        continue
                else:
                    # Default: exclude images that contain OTHER requested
                    # classes
                    other_cats = set(categories_match) - {cls_match}
                    if labels_in_image & other_cats:
                        logging.debug(
                            "Skipping %s: contains other requested classes %s",
                            sample.filepath,
                            sorted(labels_in_image & other_cats),
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
        dataset_name=args.dataset,
        splits=splits,
        per_class_limit=args.per_class_limit,
        exclude_any_non_target=args.exclude_any_non_target,
        case_insensitive=args.case_insensitive,
        verbosity=args.verbosity,
    )


if __name__ == "__main__":
    main()
