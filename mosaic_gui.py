from pathlib import Path
from PIL import Image, ImageEnhance, ImageTk
import skimage
from skimage import img_as_float
from skimage.metrics import mean_squared_error, structural_similarity
import numpy as np
from multiprocessing import Pool, cpu_count
import random
import tkinter as tk
import tkinter.filedialog
import configparser


def rmse_imgs(img1, img2):
    """ Calculate the rmse between two images """
    try:
        rmse = mean_squared_error(img_as_float(img1), img_as_float(img2))
        return rmse
    except ValueError:
        print(f'RMS issue, Img1: {img1.size[0]} {img1.size[1]}, Img2: {img2.size[0]} {img2.size[1]}')
        raise KeyboardInterrupt


class TiledImage():
    def __init__(self, target_path, folder, tile_format=(1,1), blend_factor=0, max_rep=1000, max_tiles=1000, enlargement=4):
        """ Main class used to create the tiled mosaic image """
        self.target_path = target_path
        self.folder = folder
        self.tile_format = tile_format
        self.blend_factor = blend_factor
        self.max_rep = max_rep
        self.max_tiles = max_tiles
        self.enlargement = enlargement

        self.image, self.tile_size = self.prepare_image(self.enlargement)
        print("Tile Size: ", self.tile_size)
        self.og_image = self.image.copy()
        self.tiles_x = int(self.image.width / self.tile_size[0])
        self.tiles_y = int(self.image.height / self.tile_size[1])
        self.num_tiles = self.tiles_x * self.tiles_y
        print("Using ", self.num_tiles, " tiles in total!")
        self.tile_loader = TileLoader(self.image, self.tile_size)
        self.image_loader = ImageLoader(folder, self.tile_size)
        self.scores = []
        self.data = []

    def prepare_image(self, enlargement):
        """ Calculate tile size and crop image according to enlargement size """
        img = Image.open(self.target_path)
        w = int(img.width * enlargement)
        h = int(img.height * enlargement)
        img = img.resize((w, h), Image.ANTIALIAS)

        tile_w, tile_h = self.tile_format
        tile_w1, tile_h1 = tile_w, tile_h
        mult = 1 # sensible starting point
        while ((w/tile_w1) * (h/tile_h1)) > self.max_tiles:
            tile_w1 = tile_w * mult
            tile_h1 = tile_h * mult
            mult += 1
        print(tile_w1, tile_h1)
        left = (w % tile_w1) // 2
        upper = (h % tile_h1) // 2
        right = (w % tile_w1) - left
        lower = (h % tile_h1) - upper
        img = img.crop((left, upper, w - right, h - lower))
        return img, (tile_w1, tile_h1)

    def save(self, name):
        """ Save the mosaic image """
        self.image.save(name)

    def score(self, idx):
        """ Create a list of scores for the image for all tile positions """
        img_scores = []
        image = self.image_loader[idx]
        for y in range(self.tiles_y):
            for x in range(self.tiles_x):
                img_scores.append(rmse_imgs(image, self.tile_loader[x,y]))
        return img_scores

    def spiral_order(self):
        """ sort the tiles in spiral order, to have best fitting images in the center """
        A = np.arange(self.num_tiles).reshape(self.tiles_y, self.tiles_x)
        out = []
        while (A.size):
            out.append(A[0])
            A = A[1:].T[::-1]
        return np.concatenate(out)[::-1]

    def fit_transform(self, modify=0):
        """ Main function to create the mosaic """
        self.scores = []
        self.modify=modify
        pool = Pool(processes=cpu_count())
        total_imgs = len(self.image_loader)
        self.num_used = [0] * total_imgs
        print(total_imgs , " images in the source folder")
        
        for i, score in enumerate(pool.imap(self.score, range(total_imgs), chunksize=total_imgs//60)):
            self.scores.append(score)
            percent = ("{0:.1f}").format(100 * (i / total_imgs))
            filledLength = int(100 * i // total_imgs)
            bar = '█' * filledLength + '-' * (100 - filledLength)
            print(f'\r Creating image scores: |{bar}| {percent}% ', end = '\r')

        
        self.scores = np.array(self.scores).swapaxes(0,1)
        ids = list(range(self.num_tiles))
        random.shuffle(ids)
        self.top_scores = [0] * self.num_tiles
        for tile_id in ids:
            best_score = 1000
            best_id = 0
            for i, img_score in enumerate(self.scores[tile_id]):
                if img_score < best_score and self.num_used[i] < self.max_rep:
                    best_score = img_score
                    best_id = i

            self.top_scores[tile_id] = best_id
            self.num_used[best_id] += 1
        print("")
        for i, (tile, x, y, tile_data) in enumerate(pool.imap(self.find_best_version, iter(ids), self.num_tiles//40)):
            self.paste_as_tile(tile, x, y)
            self.data.append(tile_data)
            percent = ("{0:.1f}").format(100 * (i / self.num_tiles))
            filledLength = int(100 * i // self.num_tiles)
            bar = '█' * filledLength + '-' * (100 - filledLength)
            print(f'\r Finding best versions: |{bar}| {percent}% ', end = '\r')

        print("\nTotal Mosaic Error: ", round(rmse_imgs(self.og_image, self.image), 5))
        if self.blend_factor < 0:
            self.blend_factor = (rmse_imgs(self.og_image, self.image) - 0.01) * 20
            print("blend factor: ", self.blend_factor)
        # self.og_image.putalpha(int((1-self.blend_factor) * 255))
        self.image = Image.blend(self.image, self.og_image, self.blend_factor)

    def find_best_version(self, num):
        """ If modification factor is greater than 0, find the best version for the tile """
        y = num // self.tiles_x
        x = num % self.tiles_x
        best_score = 1
        tile_data = [num, self.top_scores[num], 0, 0, 0]
        
        best_img_version = self.image_loader[self.top_scores[num]]

        if self.modify > 0:

            # Modify Color
            color = ImageEnhance.Color(best_img_version)
            for i in range(self.modify*2 + 1):
                mod_color = (i - self.modify) * 0.1
                new_img = color.enhance(1 + mod_color)
                new_rmse = rmse_imgs(new_img, self.tile_loader[x,y])
                if new_rmse < best_score:
                    best_score = new_rmse
                    best_img_version = new_img
                    tile_data[2] = mod_color  

            # Modify Brightness
            brightness = ImageEnhance.Brightness(best_img_version)
            for i in range(self.modify*2 + 1):
                mod_bright = (i - self.modify) * 0.1
                new_img = brightness.enhance(1 + mod_bright)
                new_rmse = rmse_imgs(new_img, self.tile_loader[x,y])
                if new_rmse < best_score:
                    best_score = new_rmse
                    best_img_version = new_img
                    tile_data[3] = mod_bright

            # Modify Contrast
            contrast = ImageEnhance.Contrast(best_img_version)
            for i in range(self.modify*2 + 1):
                mod_contr = (i - self.modify) * 0.1
                new_img = contrast.enhance(1 + mod_contr)
                new_rmse = rmse_imgs(new_img, self.tile_loader[x,y])
                if new_rmse < best_score:
                    best_score = new_rmse
                    best_img_version = new_img
                    tile_data[4] = mod_contr

        return best_img_version, x, y, tile_data


    def paste_as_tile(self, img, x, y):
        """ paste image onto the target image """
        self.image.paste(img, (x*self.tile_size[0], y*self.tile_size[1]))

    def create_high_res(self, scale=10):
        """ Create a new high resolution version of the mosaic """
        print("Creating high resolution image ...")
        temp = (self.tile_size[0] * scale) % self.enlargement
        wid = int(((self.tile_size[0] * scale) - temp) / self.enlargement)
        hei = int(wid * self.tile_format[1] /self.tile_format[0])
        size = wid, hei 
        og_img = self.image.resize((size[0]*self.tiles_x, size[1]*self.tiles_y), Image.ANTIALIAS)
        og_copy = og_img.copy()

        img_loader = ImageLoader(self.folder, size)
        for tile_data in self.data:
            num = tile_data[0]
            img_id = tile_data[1]
            y = num // self.tiles_x
            x = num % self.tiles_x
            img = img_loader[img_id]
            color = ImageEnhance.Color(img)
            img = color.enhance(1+tile_data[2])
            brightness = ImageEnhance.Brightness(img)
            img = brightness.enhance(1+tile_data[3])
            contrast = ImageEnhance.Contrast(img)
            img = contrast.enhance(1+tile_data[4])
            og_img.paste(img, (x*size[0], y*size[1]))
        self.image = Image.blend(og_img, og_copy, self.blend_factor)



class ImageLoader():
    """Class to load images from the source folder, applies a center crop to the images.
        Class has len() and is subscriptable"""
    def __init__(self, folder, tile_size):
        self.folder = folder
        self.images = self.collect_img_paths()
        self.tile_size = tile_size

    def collect_img_paths(self):
        """Find all images in the source folder and all subfolders"""
        imgs = []
        extensions = ['png', 'JPG', 'jpg']
        for ext in extensions:
            imgs += (list(self.folder.rglob('*.{}'.format(ext))))
        return imgs

    def crop_center(self, img):
        """Apply a center crop to the image to the desired tile format"""
        w = img.width
        h = img.height
        tile_w, tile_h = self.tile_size
        crop_w = min(w, (h/tile_h)*tile_w)
        crop_h = (crop_w/tile_w)*tile_h

        left = (w - crop_w)//2
        top = (h - crop_h)//2
        right = (w + crop_w)//2
        bottom = (h + crop_h)//2

        return img.crop((left, top, right, bottom))
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        img = self.crop_center(img)
        img = img.resize(self.tile_size, Image.ANTIALIAS)
        return img


class TileLoader():
    """ Load singular tiles from the target image 
        Get Tiles by subscription operator [] starting from top left to right bottom corner"""
    def __init__(self, target, tile_size):
        self.target = target
        self.tile_size = tile_size
    def __getitem__(self, coords):
        x, y = coords
        tile = self.target.crop((x * self.tile_size[0], y * self.tile_size[1], (x+1) * self.tile_size[0], (y+1) * self.tile_size[1]))
        return tile





if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read('mosaic_config.txt')
    ENLARGEMENT = float(config['Settings']["Enlargement"])
    NUM_TILES = int(config['Settings']['Number of tiles'])
    MAX_REPETITION = int(config['Settings']['Maximum number of repetitions'])
    BLEND_FACTOR = float(config['Settings']['Blending factor'])
    MODIFICATION = int(config['Settings']['Modification factor'])
    TILE_RATIO = tuple(int(x) for x in config['Settings']['Tile ratio'].split(':'))
    PERFORMANCE = int(config['Settings']['Performance'])
    print(TILE_RATIO)

    root = tk.Tk()
    root.withdraw()

    TARGET_PATH = tk.filedialog.askopenfilename(title='Select target image', 
            filetypes=[("image", ".jpeg"),
                        ("image", ".png"),
                        ("image", ".jpg"),
                        ("image", ".JPG")])
    SOURCE_FOLDER = tk.filedialog.askdirectory(title="Select folder with images")





    Image.MAX_IMAGE_PIXELS = None
    target_path = Path(TARGET_PATH)
    source_path = Path(SOURCE_FOLDER)

    target_tiled = TiledImage(target_path, source_path, tile_format=TILE_RATIO, blend_factor=BLEND_FACTOR, max_rep=MAX_REPETITION, enlargement=PERFORMANCE, max_tiles=NUM_TILES)
    target_tiled.fit_transform(modify=MODIFICATION)
    preview = target_tiled.image
    preview = preview.resize((int(preview.width/PERFORMANCE), int(preview.height/PERFORMANCE)), Image.ANTIALIAS)
    preview.show()

    SAVE_PATH = tk.filedialog.asksaveasfilename(title="Save mosaic image", filetypes=[("PNG", ".png")])
    if SAVE_PATH:
        target_tiled.create_high_res(ENLARGEMENT)
        target_tiled.save(SAVE_PATH)


