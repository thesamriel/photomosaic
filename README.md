# Photomosaic
A Python script to turn a target image into a mosaic consisting of multiple other images. The best images are determined by a a root_mean_squared_error function based on pixel by pixel comparison. This is an extensive task, so multiprocessing is being used, but yet the script may run multiple minutes.

### Prerequisites

```
pip install scikit-image
```

Tested with:
*   Python==3.6.9
*   Pillow==8.0.1
*   numpy==1.19.4
*   scikit-image==0.17.2

### Usage
To use the script change the global variables at the top of the script
```python
# Size of the high res mosaic: out_img = ENLARGEMENT * target_img
ENLARGEMENT = 11
# Maximum number of used tiles in the mosaic
NUM_TILES = 2000
# Maximum number each image can be used in the mosaic
MAX_REPETITION = 4
# Float between 0 and 1, how much the original image is blended over the mosaic
# negative number leads to automatic calculation of reasonable blend factor
BLEND_FACTOR = 0.15
# Integer in [0-5]. How much the image may be altered by enhancing color, brightness and contrast. 0 is no modification allowed.
MODIFICATION = 2
# Tile Format as (width, height). (1,1) is quadratic tiles
TILE_RATIO = (4, 3)

# Relative paths to the images
TARGET_PATH = "path/to/target/image.jpg"
SOURCE_FOLDER = "path/to/images/folder"
SAVE_PATH = "path/to/output/image.jpg"
```

