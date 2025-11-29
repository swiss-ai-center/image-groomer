# Image Groomer & Downloader

## Image Groomer

Simple utility to groom large set of images. While not limited to, the tools
available in the groomer are targeting the preparation of training and
evaluation sets in the context of classification tasks. It can provide simple
statistics on a set of images, detect strict or perceptual duplicates, perform
resizing and stratification into subsets (train, validation, test).

The set of images to process is defined either in one or more
paths or in one or more files. In "stat" mode, it just computes some stats
on the images. In "label" mode, it launches a simple labeling tool that
will let the user inspect the images and move them according to the class
labels defined in the label dictionary passed as argument.

When using paths, most of the tools assume a directory organisation as follows:  

```text
data_dir
├───category_a
│   └───... 
├───category_b
│   └───... 
└───category_c
    └───...
    ...
```

### Application usage

Use `python image-groomer.py --help` for help. The flag `-v` allows you to control
the verbosity level of the application (no flags for minial logs, `-v` few logs,
`-vv` more logs, etc.). The application hosts several modes:

- **stat**: The stat mode (default) computes some basic statistics on the images.
  The set of images is inferred either from the files found in dirs, or from
  list of files given in text files. Example:
  `python3 image-groomer.py -v -m stat ~/my_dir/raw_data/*_reviewed` or
  `python3 image-groomer.py ~/my-working-dir/confused_samples.txt`

- **detect_same**: With this mode, identical images are detected from the set
  of images. The detection is based on a cryptographic hash computed on the whole
  image. Removal of duplicates can be done in the tool with the flag `-rs`. There
  is also an interactive mode to visualize duplicates with the flag `-i`. Example:
  `python3 image-groomer.py -r -m detect_same my_dir/my_category > tmp.txt`.

- **detect_similar**: With this mode, similar images are computed from the set
  of images. Similarity is detected with a perceptual hash that allows for simple
  alteration of the images such as resizing, partial blurring etc. We used
  here a differential hash algorithm (dhash). The same options for removal or for
  entering into interactive mode are available (see `detect_same`).
  
- **label**: The label mode will launch a simple TkInter labeller allowing
  to quickly label images in the categories defined above. Example:
  `python3 image-groomer.py -v -m label ~/my_dir/raw_data/tmp`
  Images will be moved according to the defined category into sub-directories
  from the target (defined) directory.
  
- **make_uniform**: This mode will make the images uniform : convert all files to
  RGB, jpg compression, and optionally to a same size along width, height or both.
  An un-existing output target dir needs to be specified (different than the source dirs)
  where the data will be stored, respecting the `category_subcategory` naming convention.
  There are 3 renaming options available through argument `-t` : 'as_is', 'hash' or
  'perceptual_hash' to, respectively, use the same name, use a recomputed hash name,
  or use a recomputed perceptual hash name.
  
- **make_sets**: This mode takes source directory containing categories of images
  and split it into train, validation and test sets. The split preserves the
  class balance. Default split sizes are, respectively, 80%, 10% and 10%.
  Example source directory:

  ```text
  unsplit_data_dir
  ├───category_a
  │   └───... 
  ├───category_b
  │   └───... 
  └───category_c
      └───...
      ...
  ```

  Result after data sets creation:

  ```text
  split_data_dir
  ├───test
  │   ├───category_a
  │   ├───category_b
  │   └───category_c
  |   ...
  ├───train
  │   ├───category_a
  │   ├───category_b
  │   └───category_c
  |   ...
  └───validate
  │   ├───category_a
  │   ├───category_b
  │   └───category_c
  |   ...
  ```

### Installation with virtual environments

```sh
  python3 -m venv venv
  source ./venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
```

## Image Scraper

This script is an image bulk downloader. It downloads large numbers of
images from urls obtained with a search engine (in this case Bing Image
Search) based on a search keyword or a list of keywords. This script
automates the process of downloading and organizing large sets of images
for composing datasets, with deduplication and multi-threading support.

### How it works

1. Command-line Arguments. You can specify a search string (-s), a file with search strings inside (-f), an output directory (-o), a file prefix (-p), the number of threads, adult filter settings, filters, and a download limit.

2. Image Search. For each keyword, it queries an image search engine (in this case Bing Images), parses the result page for image URLs, and downloads them.

3. Downloading and validation. Downloads are done in parallel using threads (default 20). Each image is checked for validity and deduplicated using MD5 hashes. Images are saved with a prefix and a sanitized filename with optional prefix.

4. History. Keeps track of downloaded URLs and image hashes to avoid duplicates. Saves this history to a pickle file, and reloads it on restart.

5. Graceful Exit. On Ctrl+C, it saves the download history before exiting.

### Features

- Supports adult content filtering, applied per default, use `--no-adult-filter` to remove.
- Can apply Bing search filters (e.g., image size).
- Handles corrupted images and duplicate downloads.
- Can process multiple keywords from a file, saving each set in a separate subdirectory.
- gif files are blocked by default, use `--allow-gif` to allow them.
- Multi-threaded downloading for speed.

### Example usage

`python image-scraper.py -v -s "cats" --limit 100 --filters "+filterui:imagesize-medium"`

`python image-scraper.py -v -s "camel" --no-adult-filter -l 150 --filters "+filterui:imagesize-medium" -p "camel-image"`

`python image-scraper.py -v -s "cat" --limit 100 -o data_raw/cat --allow-gif -p "cat-2025-09"`

Note: the number of images could be accidentally higher or lower than the limit due to a weak handling of the counters in threads and the fact that downloads or files could be corrupted. This needs improvements in the code but it does the trick for bulk download.

Note: This script is designed to work with Bing Image Search. If you want to use another search engine, you will need to modify the `get_image_urls` function accordingly.

### Command-line Arguments

Use `python image-scraper.py --help` for help. The flag `-v` allows you to control the verbosity level of the application (no flags for minial logs, `-v` few logs, `-vv` more logs, etc.).

The main arguments are:

- `-s`, `--search`: Search string to query images for.
- `-f`, `--search-file`: File containing search strings line by line.
- `-o`, `--output`: Output directory.
- `-p`, `--file-prefix`: Prefix for downloaded files. If not provided, no prefix is used.
- `--no-adult-filter`: Disable adult content filtering (enabled by default).
- `--filters`: Bing search filters (e.g., `+filterui:imagesize-medium`).
- `--limit`: Maximum number of images to download per search string (default 100).
- `--threads`: Number of parallel download threads (default 20).
- `--keep-filename`: Keep original filename (default is to use md5 hash of image content).
- `--allow-gif`: Allow saving images with .gif extension (blocked by default).

## Dataset Downloaders (Open Images v7 & COCO)

Two scripts are provided to fetch curated, per-class image sets from popular
computer vision datasets via the FiftyOne dataset zoo. Each script creates one
subfolder per requested category inside a chosen output directory.

### Open Images v7 (`image-download-oidv7.py`)

Downloads images that contain at least one instance of any of the requested
classes using `only_matching=True` to minimize data transfer. The script then
applies an additional check so that images saved for a given class do not also
contain other requested classes.

Example commands:

```sh
# Download zebra and elephant images (no image will be saved twice with both)
python image-download-oidv7.py -c Zebra Elephant -o data_raw --per-class-limit 300

# Use a file listing classes (one per line) and all splits
python image-download-oidv7.py -f classes.txt -o data_raw --split all --per-class-limit 200

# Increase verbosity
python image-download-oidv7.py -c Cat Dog -o data_raw -v --per-class-limit 100
```

`classes.txt` example:

```text
Zebra
Elephant
Cat
Dog
# lines starting with # are ignored
```

Notes:

- Open Images class names are typically Title Case (e.g. `Cat`, `Dog`, `Zebra`).
- Images containing multiple requested classes are skipped for exclusivity.
- If an image has other (non-requested) labels, those may not be visible
  because we load only matching labels; this keeps downloads smaller but means
  exclusivity only covers requested categories.

### COCO (`image-download-coco.py`)

Downloads images from COCO 2014 or 2017 and saves them per requested class.
Unlike Open Images, this script loads all labels for matching images
(`only_matching=False`) so it can accurately exclude images that contain other
requested classes. An optional flag can enforce even stricter filtering.

Key options:

- `--dataset {coco-2017,coco-2014}`: Choose dataset version.
- `--split {train,validation,test,all}`: Fetch from one or both labeled splits.
- `--exclude-any-non-target`: Keep only images whose labels are exclusively the
  target class (no other objects at all).
- `--case-insensitive`: Match category names ignoring case (default enabled).

Example commands:

```sh
# Basic cat & dog download (skip images containing both)
python image-download-coco.py -c cat dog -o data_raw --per-class-limit 250

# Stricter: keep only pure zebra images (no other objects)
python image-download-coco.py -c zebra -o data_raw --exclude-any-non-target --per-class-limit 50

# Multiple classes from file, both train & validation
python image-download-coco.py -f classes.txt --split all -o data_raw --per-class-limit 100
```

Notes:

- COCO category names are lowercase (e.g. `cat`, `dog`, `zebra`). The script
  normalizes labels when `--case-insensitive` is active.
- The COCO `test` split has no annotations; filtering will yield 0 samples.
- Exclusivity logic: by default, images containing other requested classes are
  skipped; with `--exclude-any-non-target`, images containing any other object
  at all are skipped.

### General Tips

- Disk space: Large per-class limits across many categories can consume tens of
  gigabytes quickly; start with small limits.
- Reproducibility: Keep the class file under version control for dataset
  regeneration.
- Duplicates: Downstream, you can run the groomer duplicate detection modes to
  remove accidental near-duplicates across sources.
- Virtual environment: Ensure `fiftyone` and its dependencies are installed in
  the active environment before running either downloader.

## YOLO Image Verifier

A verification tool that uses YOLOv8 object detection to validate that images
contain only objects from a specified category. This is useful for cleaning
datasets by identifying images that contain the wrong objects or multiple
categories.

### YOLO verification process

1. Uses YOLOv8 to detect objects in each image
2. Verifies that only the target category is present with sufficient confidence
3. Optionally moves images that fail verification to an `_UNVERIFIED` subfolder
4. Provides detailed reports on what was detected in each image

### Requirements

This tool requires the Ultralytics library:

```sh
pip install ultralytics
```

The first run will automatically download the YOLO model weights.

### Usage examples

```sh
# Verify a single image for dogs
python verify-image-yolo.py -k dog -i image.jpg

# Verify multiple images
python verify-image-yolo.py -k cat -i img1.jpg img2.jpg img3.jpg

# Verify all images in a directory
python verify-image-yolo.py -k horse -d data_raw/horse

# Move unverified images to _UNVERIFIED subfolder
python verify-image-yolo.py -k dog -d data_raw/dog --move-unverified

# Dry run to see what would be moved without moving files
python verify-image-yolo.py -k cat -d data_raw/cat --move-unverified --dry-run

# Use a different YOLO model with custom confidence threshold
python verify-image-yolo.py -k elephant -d data_raw/elephant \
  --model yolov8m.pt --confidence 0.6 -v

# Verify images listed in a text file
python verify-image-yolo.py -k zebra -f image_list.txt --move-unverified

# Show detailed bounding box information with -vv
python verify-image-yolo.py -k dog -i image.jpg -vv

# Save cropped target boxes (for unverified images containing target objects)
python verify-image-yolo.py -k cow -d data_raw/Cattle --save-boxes --min-box-size 250 --move-unverified -vv
```

### Command-line arguments

- `-k`, `--keyword`: Target object category (required). Must be one of:
  cat, dog, bird, horse, sheep, cow, elephant, bear, zebra, giraffe, teddy bear
- `-i`, `--images`: One or more image files to verify
- `-d`, `--directory`: Directory containing images to verify
- `-f`, `--file-list`: Text file with image paths (one per line)
- `--move-unverified`: Move images that fail verification to `_UNVERIFIED`
  subfolder
- `--dry-run`: Preview what would be moved without actually moving files
- `--model`: YOLO model to use (default: yolov8n.pt). Options: yolov8n.pt,
  yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
- `--confidence`: Minimum confidence threshold (0-1, default: 0.5)
- `-v`, `--verbosity`: Increase output verbosity
  - `-v`: Show INFO level logs (verification progress)
  - `-vv`: Show DEBUG level logs with detailed bounding box information for
    each detection (class, confidence, coordinates, dimensions)
- `--save-boxes`: When an image is unverified (contains other classes) but still
  has target object detections meeting size/confidence thresholds, save cropped
  target regions into a sibling `_BOXED` directory. Crops are named
  `<original>-boxN.ext`.
- `--min-box-size`: Minimum width OR height (in pixels) for a target box to be
  saved when `--save-boxes` is used (default: 200).

### Verification logic

An image is considered verified if:

1. At least one instance of the target category is detected with sufficient
   confidence
2. No other categories are detected (ensures image purity)

An image fails verification if:

- No target category is detected
- Other object categories are present (e.g., image labeled "cat" contains a dog)

### Output

The tool provides:

- Real-time progress as each image is verified
- Detailed failure messages (what was detected instead)
- Summary statistics at the end
- Exit code 0 if all verified, 1 if any failed

### YOLO verifier notes

- The tool uses COCO dataset categories (80 classes)
- Larger models (yolov8m.pt, yolov8l.pt) provide better accuracy but are slower
- The `_UNVERIFIED` folder is created in the same directory as the images
- Use `--dry-run` first to preview changes before actually moving files
- Confidence threshold can be adjusted based on your quality requirements
- When `--save-boxes` is used, only target detections with width OR height
  >= `--min-box-size` and confidence >= `--confidence` are cropped. These
  crops are stored in a `_BOXED` directory alongside the original image.
