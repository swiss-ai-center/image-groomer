import os
import urllib.request
import re
import threading
import posixpath
import urllib.parse
import argparse
import socket
import time
import hashlib
import pickle
import signal
from PIL import Image
import io
import ssl
import logging

# config
output_dir = './download'  # default output dir
adult_filter = True        # adult filter on by default
socket.setdefaulttimeout(2)
ssl._create_default_https_context = ssl._create_stdlib_context

tried_urls = []
image_md5s = {}
in_progress = 0
downloaded_images = 0
downloaded_images_lock = threading.Lock()
urlopenheader = {
    'User-Agent': (
        'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:60.0) '
        'Gecko/20100101 Firefox/60.0'
    )
}


def download(
    pool_sema: threading.Semaphore,
    url: str,
    output_dir: str,
    output_file_prefix: str = None,
    limit: int = 100,
    verbosity: int = 0
):
    global in_progress
    global downloaded_images

    if url in tried_urls:
        return
    pool_sema.acquire()
    in_progress += 1
    path = urllib.parse.urlsplit(url).path
    # strip GET parameters from filename
    filename = posixpath.basename(path).split('?')[0]
    name, ext = os.path.splitext(filename)
    name = name[:36].strip()
    filename = output_file_prefix + '-' + name + ext

    try:
        with downloaded_images_lock:
            if downloaded_images >= limit:
                return
        request = urllib.request.Request(url, None, urlopenheader)
        image = urllib.request.urlopen(request).read()
        # Validate image using Pillow instead of imghdr
        try:
            Image.open(io.BytesIO(image)).verify()
        except Exception:
            if verbosity > 0:
                logging.warning('Invalid image format for : %s', filename)
            return

        md5_key = hashlib.md5(image).hexdigest()
        if md5_key in image_md5s:
            if verbosity > 0:
                logging.info('Image is duplicate of %s, not saving %s',
                             image_md5s[md5_key], filename)
            return

        i = 0
        while os.path.exists(os.path.join(output_dir, filename)):
            file_path = os.path.join(output_dir, filename)
            with open(file_path, 'rb') as f:
                file_md5 = hashlib.md5(f.read()).hexdigest()
            if file_md5 == md5_key:
                if verbosity > 0:
                    logging.info('Already downloaded %s, not saving',
                                 filename)
                return
            i += 1
            filename = "%s-%s-%d%s" % (output_file_prefix, name, i, ext)

        image_md5s[md5_key] = filename

        imagefile = open(os.path.join(output_dir, filename), 'wb')
        imagefile.write(image)
        imagefile.close()
        with downloaded_images_lock:
            downloaded_images += 1
        if verbosity > 1:
            logging.info('Downloaded %s to %s', url, filename)
        elif verbosity > 0:
            logging.info('Downloaded [{}] {}'.format(downloaded_images,
                                                     filename))
        tried_urls.append(url)
    except Exception as e:
        if verbosity > 0:
            logging.error('Failed to download %s: %s', url, e)
    finally:
        pool_sema.release()
        in_progress -= 1


def fetch_images_from_keyword(
        pool_sema: threading.Semaphore,
        keyword: str,
        output_dir: str,
        filters: str,
        limit: int,
        output_file_prefix: str = None,
        verbosity: int = 0):
    if verbosity > 0:
        logging.info('Fetching images for keyword: "%s"', keyword)
    global downloaded_images
    last = ''
    current = 0
    while True:
        time.sleep(0.1)

        if in_progress > 10:
            continue

        request_url = (
            'https://www.bing.com/images/async?q=' +
            urllib.parse.quote_plus(keyword) +
            '&first=' + str(current) +
            '&count=35&adlt=' + adlt +
            '&qft=' + ('' if filters is None else filters)
        )
        if verbosity > 0:
            logging.info('Requesting URL: %s', request_url)
        request = urllib.request.Request(
            request_url, None, headers=urlopenheader
        )
        response = urllib.request.urlopen(request)
        html = response.read().decode('utf8')
        links = re.findall('murl&quot;:&quot;(.*?)&quot;', html)
        try:
            if links[-1] == last:
                return
            for index, link in enumerate(links):
                with downloaded_images_lock:
                    if downloaded_images >= limit:
                        if verbosity > 0:
                            logging.info('Reached limit of %d images', limit)
                        return
                t = threading.Thread(
                    target=download,
                    args=(pool_sema, link, output_dir, output_file_prefix,
                          limit, verbosity)
                )
                t.start()
                current += 1
            last = links[-1]
        except IndexError:
            logging.error('No search results found for keyword: %s', keyword)
            return


def backup_history(signum=None, frame=None, verbosity=0):
    download_history = open(
        os.path.join(output_dir, 'download_history.pickle'), 'wb'
    )
    pickle.dump(tried_urls, download_history)
    copied_image_md5s = dict(image_md5s)
    # we are working with the copy, because length of input variable for
    # pickle must not be changed during dumping
    pickle.dump(copied_image_md5s, download_history)
    download_history.close()
    if verbosity > 0:
        logging.info('Dumped download history')
    if signum is not None:
        exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='This script is an image bulk downloader. It downloads'
        ' large numbers of images from urls obtained with as search engine'
        ' (in this case Bing Image Search) based on a search keyword or a'
        'list of keywords. This script automates the process of downloading'
        'and organizing large sets of images for composing datasets, with'
        'deduplication and multi-threading support. Example usage:'
        ' python image-scraper.py -s "cats" -o ./images -l 100')
    parser.add_argument('-v', '--verbosity', action='count', default=0,
                        help='Increase output verbosity')
    parser.add_argument('-s', '--search-string',
                        help='Keyword to search')
    parser.add_argument('-f', '--search-file',
                        help='File containing search strings line by line')
    parser.add_argument('-o', '--output', default=output_dir,
                        help='Output directory')
    parser.add_argument('-p', '--file-prefix', default=time.strftime('%Y%m%d'),
                        help='Prefix for downloaded files')
    parser.add_argument('--adult-filter', default=True,
                        action=argparse.BooleanOptionalAction,
                        help='Enable adult filter')
    parser.add_argument('--filters',
                        help='Any query based filters you want to append '
                        'when searching for images, e.g. +filterui:license-L1')
    parser.add_argument('-l', '--limit', type=int, default=100,
                        help='Limit the number of images to download')
    parser.add_argument('--threads', type=int, default=20,
                        help='Number of threads')
    args = parser.parse_args()
    # Set up logging
    if args.verbosity > 1:
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s '
                            '- %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s - %(message)s',
                            level=logging.INFO)
    # Check if at least one of the search options is provided
    if (not args.search_string) and (not args.search_file):
        logging.error('Missing search string or file with search strings')
        logging.error('Use -s or -f to specify search string or file')
        parser.print_usage(file=logging.getLogger().handlers[0].stream)
        exit(1)
    # Set up output directory
    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir_origin = output_dir
    # Set up what to do on Ctrl-C
    signal.signal(signal.SIGINT, backup_history)
    try:
        download_history = open(
            os.path.join(output_dir, 'download_history.pickle'), 'rb'
        )
        tried_urls = pickle.load(download_history)
        image_md5s = pickle.load(download_history)
        download_history.close()
    except (OSError, IOError):
        tried_urls = []
    # If the adult filter is on, we will use 'off' to disable it in the URL
    if adult_filter:
        adlt = ''
    else:
        adlt = 'off'
    if args.adult_filter:
        adlt = ''
    else:
        adlt = 'off'
    # Logging the initial configuration
    if args.verbosity > 0:
        logging.info("Starting scccrrraaapping...")
        logging.info("Output directory: %s", args.output)
        logging.info("Adult filter is set to: {}".format(args.adult_filter))
        logging.info("File prefix: %s", args.file_prefix)
        logging.info("Using filters: %s", args.filters)
        logging.info("Limiting to %d images", args.limit)
    # Set up the semaphore for limiting the number of threads
    pool_sema = threading.BoundedSemaphore(args.threads)
    if args.search_string:
        fetch_images_from_keyword(
            pool_sema,
            args.search_string,
            output_dir,
            args.filters,
            args.limit,
            args.file_prefix,
            args.verbosity
        )
    elif args.search_file:
        try:
            inputFile = open(args.search_file)
        except (OSError, IOError):
            logging.error("Couldn't open file %s", args.search_file)
            exit(1)
        for keyword in inputFile.readlines():
            output_sub_dir = os.path.join(
                output_dir_origin,
                keyword.strip().replace(' ', '_')
            )
            if not os.path.exists(output_sub_dir):
                os.makedirs(output_sub_dir)
            fetch_images_from_keyword(
                pool_sema,
                keyword,
                output_sub_dir,
                args.filters,
                args.limit,
                args.file_prefix
            )
            backup_history(verbosity=args.verbosity)
            time.sleep(10)
        inputFile.close()
