from mrcnn import utils
from PIL import Image
import os
import math
import random
import numpy as np
from DatasetUtils import *


class LeafDataset(utils.Dataset):
    def __init__(self, folder_objects=None, folder_bgs=None,
                 min_leaf=100, max_leaf=200, min_plants=10, max_plants=20,
                 leaf_angle_offset = 10, leaf_position_offset = 0,
                 image_size=512, min_scale=0.5, max_scale=1.5, min_aspect_ratio=1.2, small_leaf_percentile = 20):
        utils.Dataset.__init__(self)
        self.folder_objects = folder_objects
        self.folder_bgs = folder_bgs
        self.min_leaf = min_leaf
        self.max_leaf = max_leaf
        self.min_plants = min_plants
        self.max_plants = max_plants
        self.leaf_angle_offset = leaf_angle_offset
        self.leaf_position_offset = leaf_position_offset
        self.image_size = image_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = 1.0 / min_aspect_ratio
        self.img2 = []
        self.bg = []
        self.img_heights = []
        self.small_leaf_percentile = small_leaf_percentile

        if self.folder_objects is not None:
            self.init_objects()
        if self.folder_bgs is not None:
            self.init_bgs()

    @classmethod
    def from_config(cls, config_dict, width, height, Afolder, leaf_type):
        folder_objects = os.path.join(config_dict.get("folder_objects"), 'A' + Afolder + '_' + leaf_type)
        folder_bgs = os.path.join(config_dict.get("folder_bgs"), 'A' + Afolder)
        min_leaf = config_dict.get("min_leaf")
        max_leaf = config_dict.get("max_leaf")
        min_plants = config_dict.get("min_plants")
        max_plants = config_dict.get("max_plants")
        leaf_angle_offset = config_dict.get("leaf_angle_offset")
        leaf_position_offset = config_dict.get("leaf_position_offset")
        min_scale = config_dict.get("min_scale")
        max_scale = config_dict.get("max_scale")
        number_of_images = config_dict.get("number_of_images")
        image_size = config_dict.get("image_size")
        small_leaf_percentile = config_dict.get("small_leaf_percentile")
        leaf_dataset = cls(folder_objects, folder_bgs, min_leaf, max_leaf, min_plants, max_plants, leaf_angle_offset, leaf_position_offset, image_size, min_scale, max_scale, small_leaf_percentile=small_leaf_percentile)
        leaf_dataset.centered_leaves = config_dict.get("centered_leaves")
        leaf_dataset.small_leaf_images_ratio = config_dict.get("small_leaf_images_ratio")
        leaf_dataset.small_leaf_position_offset = config_dict.get("small_leaf_position_offset")
        leaf_dataset.small_leaves_in_image_ratio = config_dict.get("small_leaves_in_image_ratio")
        leaf_dataset.use_triads = config_dict.get("use_triads")
        leaf_dataset.img_heights = leaf_dataset.build_leaf_height_map(folder_objects)
        leaf_dataset.load_shapes(number_of_images, height, width)
        leaf_dataset.prepare()

        return leaf_dataset

    def init_objects(self):
        for root, _, files in os.walk(self.folder_objects):
            for filename in files:
                img = Image.open(os.path.join(root, filename))
                temp = img.copy()
                img.close()
                self.img2.append(temp)
        print("folder: " + self.folder_objects + " initialized")

    def init_bgs(self):
        for root, _, files in os.walk(self.folder_bgs):
            for filename in files:
                # self.bg.append(Image.open(os.path.join(root, filename)))
                tmp = Image.open(os.path.join(root, filename))
                if tmp.mode == 'RGBA':
                    tmp = tmp.convert('RGB')
                tmp = tmp.resize((self.image_size,self.image_size), resample=Image.BILINEAR)
                self.bg.append(tmp)
        _, _, files_bgs = next(os.walk(self.folder_bgs))
        self.number_of_bgs = len(files_bgs)
        print("folder: " + self.folder_bgs + " initialized")

        _, _, files_objects = next(os.walk(self.folder_objects))
        self.number_of_leafs = len(files_objects)

    def load_shapes(self, count, height, width):
        # Add classes
        self.add_class("leaves", 1, "leaf")

        num_of_small_leaf_images = math.floor(count * self.small_leaf_images_ratio)
        for i in range(count):
            # print('Image', i, end='\r')
            if self.max_plants > 1:
                bg_color, shapes = self.random_image_multiple_plants(height, width)
            else:
                bg_color, shapes = self.random_image(height, width, is_small_leaf_image=i < num_of_small_leaf_images)
            self.add_image("leaves", image_id=i, path=None, width=width, height=height, bg_color=bg_color, shapes=shapes)

    def random_image(self, height, width, is_small_leaf_image=False):
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])

        shapes = []
        boxes = []
        indexes = []
        N = random.randint(self.min_leaf, self.max_leaf)

        # AZ START INSERT
        prev_angle = random.randint(0, 360)
        min_x = math.floor(0.4 * width)
        max_x = math.floor(0.6 * width)
        min_y = math.floor(0.4 * height)
        max_y = math.floor(0.6 * height)
        x_location = random.randint(min_x, max_x)
        y_location = random.randint(min_y, max_y)
        # AZ END INSERT

        small_leafs_count = math.floor(N * self.small_leaves_in_image_ratio)
        if self.use_triads:
            leaf_idx = random.choices(range(self.number_of_leafs), k=N)
            leaf_idx = sorted(leaf_idx, key=lambda x: self.img_heights[x], reverse=True)

        for i in range(N):
            # AZ modified
            if self.centered_leaves:
                is_small_leaf = is_small_leaf_image or i >= (N - small_leafs_count) # small leaves should be last
                shape, location, scale, angle, index = self.random_shape_centered(height, width, x_location, y_location, prev_angle, i, is_small_leaf=is_small_leaf, use_triads=self.use_triads, leaf_index=leaf_idx[i])
            else:
                shape, location, scale, angle, index = self.random_shape(height, width)
            prev_angle = angle

            y, x, _ = np.asarray(self.img2[index]).shape
            shapes.append((shape, location, scale, angle, index))
            boxes.append([location[1], location[0], location[1] + y, location[0] + x])

        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.7)    # 0.5
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes

    def random_image_multiple_plants(self, height, width):
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        shapes = []
        boxes = []
        num_plants = random.randint(self.min_plants, self.max_plants)
        N = 0

        for curr_plant in range(num_plants):

            N_curr = random.randint(self.min_leaf, self.max_leaf)
            N = N + N_curr
            prev_angle = random.randint(0, 360)
            min_x = math.floor(0.2 * width)
            max_x = math.floor(0.8 * width)
            min_y = math.floor(0.2 * height)
            max_y = math.floor(0.8 * height)
            x_location = random.randint(min_x, max_x)
            y_location = random.randint(min_y, max_y)

            for _ in range(N_curr):
                if self.centered_leaves:
                    shape, location, scale, angle, index = self.random_shape_centered(height, width, x_location, y_location, prev_angle)
                else:
                    shape, location, scale, angle, index = self.random_shape(height, width)
                prev_angle = angle
                y, x, channels = np.asarray(self.img2[index]).shape
                shapes.append((shape, location, scale, angle, index))
                boxes.append([location[1], location[0], location[1] + y, location[0] + x])

        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.5)    # 0.3
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes

    def random_shape(self, height, width):
        shape = random.choice(["leaf"])
        x_location = random.randint(math.floor(0.25*width), math.floor(0.75*width))  # AZ get away from image borders
        y_location = random.randint(math.floor(0.25*height), math.floor(0.75*height))

        x_scale = random.uniform(self.min_scale, self.max_scale)
        y_scale = x_scale * random.uniform(self.min_aspect_ratio, self.max_aspect_ratio)

        angle = random.randint(0, 360)
        index = random.randint(0, self.number_of_leafs - 1)

        return shape, (x_location, y_location), (x_scale, y_scale), angle, index
    
    def get_small_leaf_index(self):
        small_leaves_indexes = []
        percentile_height = np.percentile(self.img_heights, self.small_leaf_percentile)
        for index, height in enumerate(self.img_heights):
            if height < percentile_height:
                small_leaves_indexes.append(index)

        return random.choice(small_leaves_indexes)
        
    def random_shape_centered(self, height, width, x_loc, y_loc, prev_angle, leaf_id, is_small_leaf=False, use_triads=False, leaf_index=None):
        shape = random.choice(["leaf"])

        x_scale = random.uniform(self.min_scale, self.max_scale)
        y_scale = x_scale * random.uniform(self.min_aspect_ratio, self.max_aspect_ratio)
        
        if use_triads:
            if leaf_id % 3 == 0:
                # New triad
                triad_num = leaf_id // 3
                rand_offset = 12.5 / (triad_num + 1)
                triad_angle_offset = (120 / 2**triad_num) + random.uniform(-rand_offset, rand_offset)
                angle = (prev_angle + triad_angle_offset) % 360
            else:
                angle = (prev_angle + 127.5 + random.uniform(-12.5, 12.5)) % 360
        else:    
            angle = (prev_angle + self.leaf_angle_offset + random.randint(-20, 20)) % 360   # (prev_angle + 120 + random.randint(-10, 10))
            #angle = random.randint(0, 360)

        if is_small_leaf:
            x_location = math.floor(x_loc - self.small_leaf_position_offset * math.sin(math.radians(angle)))
            y_location = math.floor(y_loc - self.small_leaf_position_offset * math.cos(math.radians(angle)))
            index = self.get_small_leaf_index()
        else:
            x_location = math.floor(x_loc - self.leaf_position_offset * math.sin(math.radians(angle))) # 64 is approx half size of leaf height in pixels
            y_location = math.floor(y_loc - self.leaf_position_offset * math.cos(math.radians(angle))) # 64   120      80  100
            index = random.randint(0, self.number_of_leafs - 1)
        
        if use_triads:
            index = leaf_index
            triad_num = leaf_id // 3
            leaf_position_offset = self.leaf_position_offset - (5*triad_num)
            percentile_height = np.percentile(self.img_heights, self.small_leaf_percentile)
            if self.img_heights[index] < percentile_height:
                leaf_position_offset = self.small_leaf_position_offset

            x_location = math.floor(x_loc - leaf_position_offset * math.sin(math.radians(angle))) 
            y_location = math.floor(y_loc - leaf_position_offset * math.cos(math.radians(angle)))
        

        return shape, (x_location, y_location), (x_scale, y_scale), angle, index

    def load_image(self, image_id):
        info = self.image_info[image_id]

        index = random.randint(0, self.number_of_bgs - 1)

        y_max, x_max, channels = np.asarray(self.bg[index]).shape

        x = random.randint(0, max(x_max - self.image_size,0))  # AZ 1024   512
        y = random.randint(0, max(y_max - self.image_size,0))  # AZ 1024   512

        area = (x, y, x + self.image_size, y + self.image_size)  # AZ 1024  512
        image = self.bg[index].crop(area)

        for shape, location, scale, angle, index in info['shapes']:
            image = self.draw_leaf(image, shape, location, scale, angle, index)
        return np.array(image)

    def draw_leaf(self, image, shape, location, scale, angle, index):
        if shape == 'leaf':
            x_location, y_location = location
            x_scale, y_scale = scale
            image = add_image(image, self.img2[index], x_location, y_location, x_scale, y_scale, angle)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)

        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        num_valid_masks = 0

        for i, (shape, location, scale, angle, index) in enumerate(info['shapes']):
            image = np.zeros([info['height'], info['width'], 3], dtype=np.uint8)
            #temp = image_to_mask(self.draw_leaf_without_transparency(image, shape, location, scale, angle, index))
            temp = self.draw_leaf_without_transparency(image, shape, location, scale, angle, index)
            temp = image_to_mask(temp)
            if np.amax(temp) == False:
                print("")
                print(" !!!!!!!!!!!!! EMPTY MASK")
                # mask[:, :, num_valid_masks] = temp[:, :]
                # num_valid_masks = num_valid_masks + 1
            mask[:, :, i] = temp[:, :]

        # if count != num_valid_masks:
        #     count = num_valid_masks
        #     mask = mask[:,:,0:num_valid_masks]

        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def draw_leaf_without_transparency(self, image, shape, location, scale, angle, index):
        """Draws a shape from the given specs."""
        if shape == 'leaf':
            x_location, y_location = location
            x_scale, y_scale = scale
            image = add_image_without_transparency(image, np.array(self.img2[index]), x_location, y_location, x_scale,
                                                   y_scale, angle)
        return image

    def build_leaf_height_map(self, folder_path):
        leaf_heights = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".png"):
                file_path = os.path.join(folder_path, filename)
                height = self.get_image_height(file_path)
                
                if height is not None:
                    leaf_heights.append(height)

        return leaf_heights
        
    def get_image_height(self, file_path):
        try:
            with Image.open(file_path) as img:
                height = img.height
                return height
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

