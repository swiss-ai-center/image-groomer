"""
Verify that images contain only objects from a specified category using YOLO.

This script uses YOLOv8 to detect objects in images and verify that they
contain only the target category (e.g., only dogs, only cats). Images that
fail verification can optionally be moved to an _UNVERIFIED subfolder.

Requirements:
- ultralytics (pip install ultralytics)

Usage examples:
    # Verify a single image
    python verify-image-yolo.py -k dog -i image.jpg

    # Verify multiple images
    python verify-image-yolo.py -k cat -i img1.jpg img2.jpg img3.jpg

    # Verify images from a directory
    python verify-image-yolo.py -k horse -d data_raw/horse

    # Move unverified images to _UNVERIFIED subfolder
    python verify-image-yolo.py -k dog -i *.jpg --move-unverified

    # Use a different YOLO model with custom confidence threshold
    python verify-image-yolo.py -k elephant -d data/ --model yolov8m.pt \
      --confidence 0.6 -v
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics is required. Install with: pip install ultralytics")
    sys.exit(1)


# Mapping of keywords to COCO class IDs (YOLOv8's predefined classes)
class_mapping = {
    'cat': 15,
    'dog': 16,
    'bird': 14,
    'horse': 17,
    'sheep': 18,
    'cow': 19,
    'elephant': 20,
    'bear': 21,
    'zebra': 22,
    'giraffe': 23,
    'teddy bear': 77,
    # Extend with more classes from COCO as needed
}

# COCO class names (for converting IDs to readable labels)
coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def verify_image(
    image_path,
    model,
    target_cls,
    keyword,
    confidence_threshold=0.5,
    verbosity=0,
    min_box_size: int = 200,
) -> Tuple[bool, List[Dict]]:
    """Verify if an image contains only the target object with sufficient
    confidence.

    Returns:
        (verified_bool, target_boxes)
        where target_boxes is a list of dicts with keys:
            'xyxy': (x1, y1, x2, y2)
            'width': w
            'height': h
            'area': w*h
            'conf': confidence
            'index': box index (1-based)
    Only boxes of the target class with width or height >= min_box_size and
    confidence >= threshold are included.
    """
    try:
        results = model(image_path)
        boxes = results[0].boxes  # Assume single image input

        if not boxes:
            print(f"Verification failed ({image_path}): No objects detected.")
            return False, []

        detected_classes = []
        target_detected = False
        target_boxes: List[Dict] = []

        # In high verbosity mode, print detailed box information
        if verbosity >= 2:
            logging.debug("\n--- Detailed Detection Info ---")
            logging.debug("Total boxes detected: %d", len(boxes))
            try:
                orig_h, orig_w = results[0].orig_shape
            except Exception:
                # Fallback if API changes
                orig_h = orig_w = None

        for idx, box in enumerate(boxes):
            cls = int(box.cls.item())
            conf = box.conf.item()
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            width = x2 - x1
            height = y2 - y1
            area = width * height
            # Print detailed box info in high verbosity mode
            if verbosity >= 2:
                if cls < len(coco_classes):
                    class_name = coco_classes[cls]
                else:
                    class_name = f"class_{cls}"

                if 'orig_h' in locals() and orig_h and orig_w:
                    rel_area = (area / (orig_w * orig_h)) * 100.0
                else:
                    rel_area = None

                if rel_area is not None:
                    logging.debug(
                        "Box %d: class='%s' (id=%d), conf=%.3f, bbox="
                        "[%.1f, %.1f, %.1f, %.1f] (w=%.1f, h=%.1f, "
                        "area=%.0f px, %.2f%% of image)",
                        idx + 1, class_name, cls, conf,
                        x1, y1, x2, y2, width, height, area, rel_area
                    )
                else:
                    logging.debug(
                        "Box %d: class='%s' (id=%d), conf=%.3f, bbox="
                        "[%.1f, %.1f, %.1f, %.1f] (w=%.1f, h=%.1f, "
                        "area=%.0f px)",
                        idx + 1, class_name, cls, conf,
                        x1, y1, x2, y2, width, height, area
                    )

            if conf >= confidence_threshold:
                detected_classes.append(cls)
                if cls == target_cls:
                    target_detected = True
                    # Collect candidate target box if size threshold satisfied
                    if width >= min_box_size or height >= min_box_size:
                        target_boxes.append(
                            {
                                "xyxy": (x1, y1, x2, y2),
                                "width": width,
                                "height": height,
                                "area": area,
                                "conf": conf,
                                "index": idx + 1,
                            }
                        )

        if verbosity >= 2:
            logging.debug(
                "Above threshold (%.2f): %d detections",
                confidence_threshold,
                len(detected_classes)
            )
            logging.debug("--- End Detection Info ---\n")

        if not target_detected:
            msg = (
                f"Verification failed ({image_path}): "
                f"'{keyword}' not detected."
            )
            print(msg)
            return False, []

        other_classes = [c for c in detected_classes if c != target_cls]
        if other_classes:
            class_names = [coco_classes[c] for c in other_classes]
            print(
                f"Verification failed ({image_path}): Detected other classes"
                f" - {class_names}."
            )
            return False, target_boxes

        return True, target_boxes
    except Exception as e:
        print(f"Verification error ({image_path}): {e}")
        return False, []


def get_image_files(paths: List[str]) -> List[Path]:
    """Collect all image files from the given paths (files or directories)."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    image_files = []

    for path_str in paths:
        path = Path(path_str)
        if path.is_file():
            if path.suffix.lower() in image_extensions:
                image_files.append(path)
        elif path.is_dir():
            for ext in image_extensions:
                image_files.extend(path.glob(f'*{ext}'))
                image_files.extend(path.glob(f'*{ext.upper()}'))
        else:
            logging.warning("Path not found: %s", path)

    return sorted(set(image_files))


def move_to_unverified(image_path: Path, dry_run: bool = False) -> bool:
    """Move an image to the _UNVERIFIED subfolder in its parent directory."""
    try:
        parent_dir = image_path.parent
        unverified_dir = parent_dir / "_UNVERIFIED"

        if not dry_run:
            unverified_dir.mkdir(exist_ok=True)
            dest_path = unverified_dir / image_path.name

            # Handle name conflicts
            counter = 1
            while dest_path.exists():
                stem = image_path.stem
                suffix = image_path.suffix
                dest_path = unverified_dir / f"{stem}_{counter}{suffix}"
                counter += 1

            shutil.move(str(image_path), str(dest_path))
            logging.info("Moved to: %s", dest_path)
        else:
            logging.info("Would move to: %s/_UNVERIFIED/%s",
                         parent_dir, image_path.name)

        return True
    except Exception as e:
        logging.error("Failed to move %s: %s", image_path, e)
        return False


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Verify that images contain only objects from a specified "
            "category using YOLO detection."
        )
    )

    parser.add_argument(
        "-k", "--keyword",
        required=True,
        help="Target object category (e.g., dog, cat, horse)",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i", "--images",
        nargs="+",
        help="Image file(s) to verify",
    )
    input_group.add_argument(
        "-d", "--directory",
        help="Directory containing images to verify",
    )
    input_group.add_argument(
        "-f", "--file-list",
        help="Text file containing image paths (one per line)",
    )

    parser.add_argument(
        "--move-unverified",
        action="store_true",
        help="Move images that fail verification to _UNVERIFIED subfolder",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without actually moving files",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model to use (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for detections (default: 0.5)",
    )
    parser.add_argument(
        "--save-boxes",
        action="store_true",
        help=(
            "When an image is unverified but contains target objects, save "
            "cropped target boxes meeting size threshold into a _BOXED folder"
        ),
    )
    parser.add_argument(
        "--min-box-size",
        type=int,
        default=200,
        help=(
            "Minimum width OR height in pixels for a target box to be saved "
            "(default: 200)"
        ),
    )
    parser.add_argument(
        "-v", "--verbosity",
        action="count",
        default=0,
        help="Increase output verbosity (-v, -vv)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    log_level = logging.WARNING
    if args.verbosity == 1:
        log_level = logging.INFO
    elif args.verbosity >= 2:
        log_level = logging.DEBUG

    logging.basicConfig(
        format="%(levelname)s - %(message)s",
        level=log_level,
    )

    # Validate keyword
    keyword = args.keyword.lower()
    if keyword not in class_mapping:
        available = ", ".join(sorted(class_mapping.keys()))
        logging.error(
            "Unknown keyword '%s'. Available: %s", args.keyword, available
        )
        sys.exit(1)

    target_cls = class_mapping[keyword]
    logging.info(
        "Target category: %s (COCO class ID: %d)", keyword, target_cls
    )

    # Collect image files
    if args.images:
        image_paths = args.images
    elif args.directory:
        image_paths = [args.directory]
    else:  # args.file_list
        try:
            with open(args.file_list, 'r', encoding='utf-8') as f:
                image_paths = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith('#')
                ]
        except Exception as e:
            logging.error("Failed to read file list: %s", e)
            sys.exit(1)

    image_files = get_image_files(image_paths)

    if not image_files:
        logging.error("No image files found")
        sys.exit(1)

    logging.info("Found %d image(s) to verify", len(image_files))

    # Load YOLO model
    logging.info("Loading YOLO model: %s", args.model)
    try:
        model = YOLO(args.model)
    except Exception as e:
        logging.error("Failed to load YOLO model: %s", e)
        sys.exit(1)

    # Verify images
    verified_count = 0
    unverified_count = 0
    error_count = 0

    for image_path in image_files:
        logging.info("\nVerifying: %s", image_path)
        is_verified, target_boxes = verify_image(
            str(image_path),
            model,
            target_cls,
            keyword,
            confidence_threshold=args.confidence,
            verbosity=args.verbosity,
            min_box_size=args.min_box_size,
        )

        if is_verified:
            verified_count += 1
            logging.info("✓ Verified: %s", image_path.name)
        else:
            unverified_count += 1
            # Save cropped target boxes if requested and present
            if args.save_boxes and target_boxes:
                boxed_dir = image_path.parent / "_BOXED"
                boxed_dir.mkdir(exist_ok=True)
                try:
                    with Image.open(image_path) as im:
                        for bidx, box_info in enumerate(target_boxes, start=1):
                            x1, y1, x2, y2 = box_info["xyxy"]
                            # Clamp and convert to int
                            x1 = int(max(0, round(x1)))
                            y1 = int(max(0, round(y1)))
                            x2 = int(min(im.width, round(x2)))
                            y2 = int(min(im.height, round(y2)))
                            if x2 <= x1 or y2 <= y1:
                                continue
                            crop = im.crop((x1, y1, x2, y2))
                            stem = image_path.stem
                            ext = image_path.suffix
                            out_name = f"{stem}-box{bidx}{ext}"
                            out_path = boxed_dir / out_name
                            dup_i = 1
                            while out_path.exists():
                                out_name = f"{stem}-box{bidx}-{dup_i}{ext}"
                                out_path = boxed_dir / out_name
                                dup_i += 1
                            crop.save(out_path)
                            logging.info(
                                "Saved box %d crop %s (w=%d h=%d area=%d)",
                                bidx,
                                out_name,
                                box_info["width"],
                                box_info["height"],
                                box_info["area"],
                            )
                except Exception as e:
                    logging.warning(
                        "Failed saving cropped boxes for %s: %s",
                        image_path.name,
                        e,
                    )
            if args.move_unverified:
                if move_to_unverified(image_path, dry_run=args.dry_run):
                    logging.info("✗ Unverified and moved: %s", image_path.name)
                else:
                    error_count += 1
            else:
                logging.info("✗ Unverified: %s", image_path.name)

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total images:      {len(image_files)}")
    print(f"Verified:          {verified_count}")
    print(f"Unverified:        {unverified_count}")
    if args.move_unverified:
        print(f"Move errors:       {error_count}")
    if args.dry_run:
        print("\n(Dry run - no files were actually moved)")
    print("=" * 60)

    # Exit with appropriate code
    if unverified_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
