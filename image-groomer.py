from typing import List
import os
import shutil
import argparse
import json
import glob
import logging
import random
import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image
import hashlib
import imagehash

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

# you may want to adapt the following meta variables to your usage
allowed_ext = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
default_labels = {'category-1': '1',
                  'category-2': '2',
                  'category-3': '3',
                  'trash': '-',
                  'keep and next': 'k'}


class ImageMeta():
    """
    An ImageMeta object is here storing information on the image itself and
    meta information on its ground truth (if available) and on the hypothesis
    out of a detection system (also if available).
    """

    def __init__(self, filename: str, category: str = None,
                 subcategory: str = None, verbosity: int = 0):
        self.filename = filename
        if not os.path.isfile(filename):
            logging.error('File [{}] does not exist. Check format of file '
                          'lists, should be one valid filename per line '
                          'without other trailing info.'.format(filename))
            exit(1)
        self.category = category
        self.subcategory = subcategory
        self.dir, self.name_ext = os.path.split(filename)
        # get file name without extension
        self.name = os.path.splitext(self.name_ext)[0]
        if '.' in self.name and verbosity > 1:
            logging.warning('file name contains a dot "." after ext removed: '
                            + '[{}]'.format(self.filename))
        if self.category is None:
            # try to infer category from filename (last dir of filename)
            # get last part of target_dir
            unused, self.category = os.path.split(self.dir)
            if '_' in self.category:
                self.subcategory = self.category[self.category.find('_') + 1:]
                self.category = self.category[0:self.category.find('_')]
                self.cat_sub = self.category + '_' + self.subcategory
            else:
                self.cat_sub = self.category
        im = Image.open(self.filename)
        self.format = im.format
        self.width, self.height = im.size
        # L for grey-scale, RGB for 3-channels color images
        self.mode = im.mode
        if self.height >= self.width:
            self.orientation = 'portrait'
        else:
            self.orientation = 'landscape'
        self.hash = None
        self.dhash = None
        if verbosity > 1:
            logging.info("name        = {}".format(self.name))
            logging.info("format      = {}".format(self.format))
            logging.info("category    = {}".format(self.category))
            logging.info("subcategory = {}".format(self.subcategory))
            logging.info("w, h        = ({},{})".format(self.width,
                                                        self.height))
            logging.info("orientation = {}".format(self.orientation))
            logging.info("mode        = {}".format(self.mode))
        del im

    def compute_hash(self):
        '''
        Compute a cryptographic hash for exact match comparisons.
        :return: The hash value
        '''
        with open(self.filename, 'rb') as f:
            self.hash = hashlib.md5(f.read()).hexdigest()
        return self.hash

    def compute_dhash(self):
        '''
        Compute a perceptual hash. See https://pypi.org/project/ImageHash/
        for more information.
        :return: The d-hash value
        '''
        im = Image.open(self.filename)
        self.dhash = imagehash.dhash(im)
        del im
        return self.dhash

    def build_dict(self):
        data = {'name': self.name, 'format': self.format,
                'category': self.category, 'subcategory': self.subcategory,
                'size': (self.width, self.height), 'mode': self.mode}
        if self.hash is not None:
            data['hash'] = self.hash
        return data

    def __str__(self):
        return json.dumps(self.build_dict())


def get_all_images_in_dir(root_dir: str, recursive: bool = True,
                          verbosity: int = 0) -> List[str]:
    if not root_dir.endswith('/'):
        root_dir += '/'
    filelist = []
    if recursive:
        pattern = '**/*'
    else:
        pattern = '*'
    for filename in glob.iglob(root_dir + pattern, recursive=recursive):
        name, extension = os.path.splitext(filename)
        if extension in allowed_ext:
            if verbosity > 1:
                logging.info('Found [{}]'.format(filename))
            filelist.append(filename)
        else:
            if verbosity > 1:
                logging.info('Discarded [{}]'.format(filename))
    return filelist


def move_file(from_file: str, to_file: str, create_target_dirs: bool = False,
              keep_original: bool = False, verbosity: int = 0):
    if create_target_dirs:
        # we check here if the target dir exist, if not create it
        target_dir, target_file = os.path.split(to_file)
        if not os.path.isdir(target_dir):
            if verbosity > 0:
                logging.info("creating dir [{}]".format(target_dir))
            os.makedirs(target_dir)
    if verbosity > 1:
        logging.info("moving from [{}] to [{}]".format(from_file, to_file))
    if keep_original:
        shutil.copyfile(from_file, to_file)
    else:
        os.rename(from_file, to_file)


def save_jpeg_image(img, filename, create_target_dirs: bool = False,
                    verbosity: int = 0):
    if create_target_dirs:
        # we check here if the target dir exist, if not create it
        target_dir, target_file = os.path.split(filename)
        if not os.path.isdir(target_dir):
            if verbosity > 0:
                logging.info("creating dir [{}]".format(target_dir))
            os.makedirs(target_dir)
    if verbosity > 1:
        logging.info("saving image to [{}]".format(filename))
    # image quality for saving should potentially be checked here,
    # default to 75, maybe 95 is better
    img.save(filename, 'JPEG')


def get_label_for_val(val, labels):
    for key, value in labels.items():
        if val == value:
            return key
    return None


# Print iterations progress
def print_progress_bar(iteration, total, prefix: str = 'Progress:',
                       suffix: str = 'Complete', decimals: int = 1,
                       length: int = 50, fill: str = '█',
                       printEnd: str = '\r'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent
                                  complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def compute_stat(filelist, verbosity: int = 0):
    if verbosity > 0:
        logging.info('Computing stats on {} images'.format(len(filelist)))
    n_files = len(filelist)
    step = n_files // 100
    filelist.sort()
    # get all meta info
    images = []
    for i, filename in enumerate(filelist):
        if verbosity > 0 and n_files > 1000:
            if i % step == 0 or i == n_files-1:
                print_progress_bar(i+1, n_files)
        images.append(ImageMeta(filename, verbosity=verbosity))
    # images = [
    #    ImageMeta(filename, verbosity=verbosity)
    #    for filename in filelist
    # ]
    # now do some stats
    categories = {}
    subcategories = {}
    formats = {}
    modes = {}
    sizes = {}
    orientations = {}
    for image in images:
        increment_dictionary(categories, image.category)
        if image.subcategory is not None:
            increment_dictionary(subcategories, image.category + '_' +
                                 image.subcategory)
        increment_dictionary(formats, image.format)
        increment_dictionary(modes, image.mode)
        increment_dictionary(sizes, '({},{})'.format(image.width,
                                                     image.height))
        increment_dictionary(orientations, image.orientation)
    logging.info("categories      : " + json.dumps(categories))
    if len(subcategories) > 0:
        logging.info("subcategories   : " + json.dumps(subcategories))
    logging.info("formats         : " + json.dumps(formats))
    logging.info("modes           : " + json.dumps(modes))
    logging.info("orientations    : " + json.dumps(orientations))
    logging.info("different sizes : {}".format(len(sizes)))
    if len(sizes) < 10:
        logging.info("sizes           : " + json.dumps(sizes))


def label_images(filelist, outputdir: str = '', labels: str = '',
                 verbosity: int = 0):
    if labels == '':
        labels = default_labels
    else:
        labels = json.loads(labels)
    if verbosity > 0:
        logging.info('Label images using the following labels')
        for label in labels:
            logging.info("[{}] with shortcut key '{}'".format(label,
                                                              labels[label]))

    def key_pressed(event):
        pressed_key = str(event.char)
        if verbosity > 1:
            print("pressed [{}] for file {}".format(pressed_key,
                                                    image_file_basename))
        label = get_label_for_val(pressed_key, labels)
        if outputdir != '':
            image_path = outputdir
        if label is not None:
            if label == 'quit':
                root_win.destroy()
                exit(0)
            if label != 'keep and next':
                move_file(image_file, image_path + '/' + label + '/' +
                          image_file_basename, create_target_dirs=True,
                          verbosity=verbosity)
            root_win.destroy()
        else:  # invalid pressed_key
            messagebox.showinfo("Grooomer",
                                "Invalid key [{}]".format(pressed_key))

    for i, image_file in enumerate(filelist):
        # before anything check if image file exists
        if not os.path.isfile(image_file):
            logging.warning('File [{}] does not exist. Check format of file '
                            'lists, should be one valid filename per line '
                            'without other trailing info.'.format(image_file))
            continue
        root_win = tk.Tk()
        image_path, image_file_basename = os.path.split(image_file)
        root_win.title(image_file_basename + f' {i+1}/{len(filelist)}')
        root_win.geometry("+300+300")
        root_win.focus_set()
        root_win.bind('<Key>', key_pressed)
        img_pil = Image.open(image_file)
        img_pil.thumbnail((800, 800), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img_pil)
        tklabel = tk.Label(root_win, image=img)
        tklabel.pack()
        next_button = tk.Button(root_win, text='Next',
                                command=root_win.destroy)
        next_button.pack()
        for label in labels:
            tk.Label(root_win,
                     text='{} -> {}'.format(label, labels[label])).pack()
        tk.Label(root_win, text=f'{image_file}').pack()
        tk.mainloop()


def detect_duplicate(filelist, mode: str, remove_similar: bool = False,
                     interactive: bool = True, verbosity: int = 0):
    if verbosity > 0:
        logging.info('Detecting duplicates in {} mode'.format(mode))
    # get all meta info
    images = [
        ImageMeta(filename, verbosity=verbosity)
        for filename in filelist
    ]
    if verbosity > 0:
        logging.info('Computing hashes for {}'.format(mode))
        logging.info('Looking into {} images'.format(len(filelist)))
    hashdic = {}
    count_similar = 0
    similar_images = []   # list of couples of images found to be similar
    print_progress_bar(0, len(filelist))
    for i, image in enumerate(images):
        if mode == 'detect_same':
            hash_value = image.compute_hash()
        else:  # detect_similar with perceptual hash
            hash_value = image.compute_dhash()
        if hash_value in hashdic:   # found a similar image
            if verbosity > 1:
                print("TO_SCRIPT_1: open " + image.filename + " " +
                      hashdic[hash_value].filename)
                print("TO_SCRIPT_2: rm {}".format(image.filename))
            else:
                if verbosity > 0:
                    logging.info('found same image with {}: {} <-> {}'.format(
                        mode, image.filename, hashdic[hash_value].filename))
            if remove_similar:
                if verbosity > 0:
                    logging.info('remove similar: {}'.format(image.filename))
                os.remove(image.filename)
            similar_images.append((image.filename,
                                   hashdic[hash_value].filename))
            count_similar += 1
        else:
            hashdic[hash_value] = image
        print_progress_bar(i + 1, len(filelist))
    logging.info('Found {} similar images'.format(count_similar))
    if remove_similar:
        logging.info('Removed {} similar images'.format(count_similar))
    if interactive and not remove_similar and count_similar > 0:
        if args.verbosity > 0:
            logging.info('Entering interactive mode')
        for similar_image_path_tuple in similar_images:
            if args.verbosity > 0:
                logging.info('Treating: ' + str(similar_image_path_tuple))
            root_win = tk.Tk()
            image_path_1, image_file_basename_1 = (
                os.path.split(similar_image_path_tuple[0])
            )
            category_1 = image_path_1.split(os.sep)[-1]
            image_path_2, image_file_basename_2 = (
                os.path.split(similar_image_path_tuple[1]))
            category_2 = image_path_2.split(os.sep)[-1]
            root_win.title('Duplicates')
            root_win.geometry("+300+300")
            root_win.focus_set()
            # image 1
            tk.Label(root_win, text=image_file_basename_1).grid(row=0,
                                                                column=0)
            img_pil_1 = Image.open(similar_image_path_tuple[0])
            img_pil_1.thumbnail((640, 640), Image.Resampling.LANCZOS)
            img_1 = ImageTk.PhotoImage(img_pil_1)
            tklabel_1 = tk.Label(root_win, image=img_1)
            tklabel_1.grid(row=1, column=0)
            # image 2
            tk.Label(root_win, text=image_file_basename_2).grid(row=0,
                                                                column=2)
            img_pil_2 = Image.open(similar_image_path_tuple[1])
            img_pil_2.thumbnail((640, 640), Image.Resampling.LANCZOS)
            img_2 = ImageTk.PhotoImage(img_pil_2)
            tklabel_2 = tk.Label(root_win, image=img_2)
            tklabel_2.grid(row=1, column=2)
            # bottom rows
            tk.Label(root_win, text=category_1).grid(row=2, column=0)
            tk.Label(root_win, text=category_2).grid(row=2, column=2)

            def next_command(event=None):
                root_win.destroy()

            def remove_1(event=None):
                os.remove(similar_image_path_tuple[0])
                if args.verbosity > 0:
                    logging.info('removed' + similar_image_path_tuple[0])
                root_win.destroy()

            def remove_2(event=None):
                os.remove(similar_image_path_tuple[1])
                if args.verbosity > 0:
                    logging.info('removed' + similar_image_path_tuple[1])
                root_win.destroy()
            tk.Button(root_win, text='Remove [1]',
                      command=remove_1).grid(row=3, column=0)
            tk.Button(root_win, text='Next',
                      command=next_command).grid(row=3, column=1)
            tk.Button(root_win, text='Remove [2]',
                      command=remove_2).grid(row=3, column=2)
            root_win.bind('1', remove_1)
            root_win.bind('2', remove_2)
            root_win.bind('<Return>', next_command)
            tk.mainloop()


def make_uniform(filelist, outputdir: str, width: int = -1, height: int = -1,
                 namestrategy: str = None, verbosity: int = 0):
    # first check that the outputdir is not existing
    if os.path.isdir(outputdir):
        logging.error('output dir [{}] should not be existing'
                      .format(outputdir))
        exit(1)
    # then create it
    os.makedirs(outputdir)
    if verbosity > 0:
        logging.info('created output dir [{}]'.format(outputdir))
        logging.info('converting to width  = {}'.format(width))
        logging.info('converting to height = {}'.format(height))
    filelist.sort()
    # check resize info
    resize_both = False
    resize_width = False
    resize_height = False
    if width != -1 and height != -1:
        resize_both = True
    elif width == -1 and height != -1:
        resize_height = True
    elif width != -1 and height == -1:
        resize_width = True
    # get all meta info
    images = [ImageMeta(filename, verbosity=verbosity) for filename
              in filelist]
    for image in images:
        # define filename
        target_name = ""
        if namestrategy == 'as_is':
            target_name = image.name
        elif namestrategy == 'perceptual_hash':
            target_name = str(image.compute_dhash())
        elif namestrategy == 'hash':
            target_name = image.compute_hash()
        else:
            logging.error('namestrategy argument to make_uniform() is invalid')
            exit(1)
        # if no subcategory, we just use category
        if image.subcategory is None:
            target_file_name = outputdir + os.sep + image.category + os.sep \
                + target_name + '.jpg'
        else:
            target_file_name = outputdir + os.sep + image.category + '_' \
                + image.subcategory + os.sep + target_name + '.jpg'
        if verbosity > 1:
            logging.info('target file is {}'.format(target_file_name))
        im = Image.open(image.filename)
        # convert if needed to RGB
        if image.mode != 'RGB':
            im = im.convert('RGB')
            if verbosity > 1:
                logging.info('converted image mode from {} to RGB: {}'
                             .format(image.mode, target_file_name))
        # resizing if needed
        if resize_both:
            im = im.resize((width, height), Image.Resampling.LANCZOS)
        elif resize_width:
            im = im.resize((width, int(width * image.height / image.width)),
                           Image.Resampling.LANCZOS)
        elif resize_height:
            im = im.resize((int(height * image.width / image.height), height),
                           Image.Resampling.LANCZOS)
        save_jpeg_image(im, target_file_name, create_target_dirs=True,
                        verbosity=verbosity)


def increment_dictionary(dico, key):
    if key not in dico:
        dico[key] = 0
    dico[key] += 1


def make_train_validation_test_sets(filelist, outputdir: str,
                                    train_size: float = 0.8,
                                    val_size: float = 0.1,
                                    test_size: float = 0.1,
                                    nperclass: int = -1,
                                    seed: int = -1, verbosity: int = 0):
    # check that split values are correct
    if abs(1.0 - (train_size + val_size + test_size)) > 0.001:
        logging.error("split sizes not correct, should sum up to 1.0")
        exit(1)
    # check that the outputdir is not existing
    if os.path.isdir(outputdir):
        logging.error('output dir [{}] should not be existing'
                      .format(outputdir))
        exit(1)
    # then create it
    os.makedirs(outputdir)
    if verbosity > 0:
        logging.info('created output dir [{}]'.format(outputdir))
    compute_stat(filelist, verbosity)
    # get all meta info
    images = [ImageMeta(filename, verbosity=verbosity) for filename
              in filelist]
    random.shuffle(images)
    # compose lists of images per categories
    categories = {}
    if verbosity > 0 and nperclass != -1:
        logging.info('limiting to max {} images per classes'.format(nperclass))
    for image in images:
        if image.cat_sub not in categories:
            categories[image.cat_sub] = []
        # check if there is a limit on the number of image per class
        # (-1 means no limit);
        # if we reached the limit then we should stop appending
        if nperclass == -1 or len(categories[image.cat_sub]) < nperclass:
            categories[image.cat_sub].append(image)
    for category in categories:
        if verbosity > 0:
            logging.info('category[{}] contains {} images'
                         .format(category, len(categories[category])))
    # create sub-dirs train, validate and test
    os.makedirs(outputdir + os.path.sep + 'train')
    os.makedirs(outputdir + os.path.sep + 'validate')
    os.makedirs(outputdir + os.path.sep + 'test')
    # set the seed if needed
    if seed != -1:
        random.seed(seed)
        if verbosity > 0:
            logging.info('seed value is set to : {}'.format(seed))
    for category in categories:
        # list of images in this category
        category_images = categories[category]
        n_validation = int(len(category_images) * val_size)
        n_test = int(len(category_images) * test_size)
        n_train = len(category_images) - n_validation - n_test
        if verbosity > 0:
            logging.info('category[{}] split with train={}, '
                         'validation={}, test={}'
                         .format(category, n_train, n_validation, n_test))
        # now move the images
        for i in range(len(category_images)):
            # pick a random image in the list for the current category
            index = random.randint(0, len(category_images) - 1)
            image = category_images[index]
            if i < n_validation:
                target_dir = 'validate'
            elif i < n_validation + n_test:
                target_dir = 'test'
            else:
                target_dir = 'train'
            target_filename = outputdir + os.path.sep + target_dir + \
                os.path.sep + category + os.path.sep + image.name_ext
            if verbosity > 1:
                logging.info('moving image from {} to {}'
                             .format(image.filename, target_filename))
            # now remove it from the list so that it is not picked again
            category_images.pop(index)
            # finally move it
            move_file(image.filename, target_filename,
                      create_target_dirs=True, keep_original=True,
                      verbosity=verbosity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Simple utility to groom large set of images defined '
        'either in one or more paths or in one or more files. In "stat" '
        'mode, it just computes some stats on the images. In "label" '
        'mode, it launches a simple labeling tool that will let the user '
        'inspect the images and move them according to the class labels '
        'defined in the label dictionary passed as argument.')
    parser.add_argument('-v', '--verbosity', action='count', default=0,
                        help='increase output verbosity')
    parser.add_argument('-m', '--mode', type=str, default='stat',
                        choices=['stat', 'label', 'detect_same',
                                 'detect_similar', 'make_uniform',
                                 'make_sets'],
                        help='mode can be "stat", "label", "detect_same", '
                             '"detect_similar", "make_uniform" or "make_sets"')
    parser.add_argument('-t', '--targetname', type=str, default='as_is',
                        choices=['as_is', 'hash', 'perceptual_hash'])
    parser.add_argument('-r', '--recursive', action='store_true',
                        default=False)
    parser.add_argument('-o', '--outputdir', type=str, default='groomed',
                        help='where groomed images will be')
    parser.add_argument('-wi', '--width', type=int, default=-1,
                        help='max width')
    parser.add_argument('-he', '--height', type=int, default=-1,
                        help='max height')
    parser.add_argument('-rs', '--removesim', action='store_true',
                        default=False, help='remove similar files')
    parser.add_argument('-i', '--interactive', action='store_true',
                        default=False,
                        help='interactive mode (e.g. for detect_same mode)')
    parser.add_argument('-l', '--labels', type=str, default='',
                        help='labels for the label mode, pass in as JSON '
                        'object')
    parser.add_argument('-n', '--nperclass', type=int, default=-1,
                        help='max number of images per class')
    parser.add_argument('-s', '--shuffle', action='store_true', default=False,
                        help='shuffle the input image set before any '
                             'operation')
    parser.add_argument('-se', '--seed', type=int, default=-1,
                        help='seed value for random set generation')
    parser.add_argument('root_paths', metavar='PATHS', type=str, nargs='+',
                        help='root paths or files from which we grrrrooom')
    args = parser.parse_args()
    if args.verbosity > 1:
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s '
                            '- %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s - %(message)s',
                            level=logging.INFO)
    if args.verbosity > 0:
        logging.info("Starting grrrooooming...")
        logging.info("Reading files from {}".format(args.root_paths))
        logging.info("Recursive mode is {}".format(args.recursive))
    # Get the list of files, either from dirs or from the content of files
    # given as last arguments. The objective is to populate filelist with
    # all the images to be groomed.
    filelist = []
    for root_dir in args.root_paths:
        if os.path.isdir(root_dir):
            root_dir_filelist = \
                get_all_images_in_dir(root_dir, recursive=args.recursive,
                                      verbosity=args.verbosity)
            if args.verbosity > 0:
                logging.info("Found {} images in dir [{}]"
                             .format(len(root_dir_filelist), root_dir))
            filelist.extend(root_dir_filelist)
        elif os.path.isfile(root_dir):
            with open(root_dir) as f:
                root_dir_filelist = f.readlines()
                root_dir_filelist = [x.strip() for x in root_dir_filelist]
                if args.verbosity > 0:
                    logging.info("Found {} images in file [{}]"
                                 .format(len(root_dir_filelist), root_dir))
                filelist.extend(root_dir_filelist)
            if args.recursive:
                logging.error('Recursive mode is not allowed with files')
                exit(1)
        else:
            logging.error('Path [{}] is not a valid dir or file'.format(
                root_dir))
            exit(1)
    logging.info('Found {} images'.format(len(filelist)))
    if len(filelist) == 0:
        logging.info('No image found in the given paths or files. Exiting.')
        exit(1)
    if args.shuffle:
        logging.info('Shuffling the list of images')
        random.shuffle(filelist)
    if args.mode == 'stat':
        compute_stat(filelist,
                     verbosity=args.verbosity)
    if args.mode == 'label':
        label_images(filelist,
                     outputdir=args.outputdir,
                     labels=args.labels,
                     verbosity=args.verbosity)
    if args.mode == 'detect_same' or args.mode == 'detect_similar':
        detect_duplicate(filelist,
                         args.mode,
                         remove_similar=args.removesim,
                         interactive=args.interactive,
                         verbosity=args.verbosity)
    if args.mode == 'make_uniform':
        make_uniform(filelist,
                     args.outputdir,
                     args.width, args.height,
                     namestrategy=args.targetname,
                     verbosity=args.verbosity)
    if args.mode == 'make_sets':
        make_train_validation_test_sets(filelist,
                                        args.outputdir,
                                        nperclass=args.nperclass,
                                        seed=args.seed,
                                        verbosity=args.verbosity)
