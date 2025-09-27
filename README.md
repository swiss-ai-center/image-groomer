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
