import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import requests
import torch
import torchvision.transforms as T
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from typing import Union, Any

class BoundingBox:
    """
    A class representing a bounding box with a score.
    
    :param x1: The x-coordinate of the top-left corner.
    :param y1: The y-coordinate of the top-left corner.
    :param x2: The x-coordinate of the bottom-right corner.
    :param y2: The y-coordinate of the bottom-right corner.
    :param score: The confidence score of the bounding box.
    """

    def __init__(self, x1:float, y1:float, x2:float, y2:float, score:float=0.0):
        """
        Initialize a BoundingBox instance.
        
        :param x1: The x-coordinate of the top-left corner.
        :param y1: The y-coordinate of the top-left corner.
        :param x2: The x-coordinate of the bottom-right corner.
        :param y2: The y-coordinate of the bottom-right corner.
        :param score: The confidence score of the bounding box. Defaults to 0.0.
        """
        self.coords = (x1, y1, x2, y2)
        self.score = score

    def __getitem__(self, index: int) -> float:
        """
        Allows accessing the bounding box coordinates like a list.
        
        :param index: The index of the coordinate.
            
        :return: The coordinate at the specified index.
        """
        return self.coords[index]

    def __setitem__(self, index: int, value: float):
        """
        Allows setting the bounding box coordinates like a list.
        
        :param index: The index of the coordinate to set.
        :param value: The new value for the coordinate.
        """
        self.coords[index] = value

    def __repr__(self) -> str:
        """
        Returns a string representation of the BoundingBox instance.
        
        :return: The string representation of the BoundingBox.
        """
        return f"BoundingBox({self.coords[0]}, {self.coords[1]}, {self.coords[2]}, {self.coords[3]}, score={self.score})"

    def normalize(self, image_width: Union[float, int], image_height: Union[float, int]):
        """
        Normalizes the bounding box coordinates with respect to the image dimensions.
        
        :param image_width: The width of the image.
        :param image_height: The height of the image.
            
        :return: A new instance of BoundingBox with normalized coordinates.
        """
        x1, y1, x2, y2 = self.coords
        return BoundingBox(x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height, self.score)
    
class BoundingBoxCluster:
    """Class representing a cluster of bounding boxes."""

    def __init__(self, boxes: list[BoundingBox]):
        self.boxes = boxes
        self.graph = {i: set() for i in range(len(boxes))}
        self.build_graph()
    
    def is_overlapping(self, box1: BoundingBox, box2: BoundingBox) -> bool:
        """Check if two boxes (x1, y1, x2, y2) overlap."""
        return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])
    
    def build_graph(self):
        """Build a graph based on overlaps between bounding boxes in `self.boxes`."""
        for i in range(len(self.boxes)):
            for j in range(i + 1, len(self.boxes)):
                if self.is_overlapping(self.boxes[i], self.boxes[j]):
                    self.graph[i].add(j)
                    self.graph[j].add(i)
    
    def find_connected_components(self):
        """Find connected components (clusters) in `self.graph`."""

        visited = set()
        components = []
        
        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for node in range(len(self.boxes)):
            if node not in visited:
                component = []
                dfs(node, component)
                components.append(component)
        
        return components
    
    def merge_boxes(self, component) -> BoundingBox:
        """Merge all boxes in a connected component."""
        x1 = min(self.boxes[node][0] for node in component)
        y1 = min(self.boxes[node][1] for node in component)
        x2 = max(self.boxes[node][2] for node in component)
        y2 = max(self.boxes[node][3] for node in component)
        score = max(self.boxes[node].score for node in component)
        return BoundingBox(x1, y1, x2, y2, score)
    
    def get_merged_boxes(self):
        """Get merged bounding boxes for each connected component."""
        components = self.find_connected_components()
        return [self.merge_boxes(component) for component in components]

def get_preprocessed_image(pixel_values: torch.FloatTensor) -> Image.Image:
    """
    Preprocesses pixel values of an image to pad a non-square image into a square.

    :param pixel_values: tensor of pixel values.
    :return: padded image.
    """
    pixel_values = pixel_values.squeeze().numpy()
    unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    unnormalized_image = Image.fromarray(unnormalized_image)
    return unnormalized_image


def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)

def draw_entity_boxes_on_image(image: Union[Image.Image, str, torch.Tensor], entities: list[Any], show=False, save_path=None):
    """
    Helper function to draw bounding boxes over a given image. Adapted from Kosmos-2 code at https://huggingface.co/microsoft/kosmos-2-patch14-224.

    :param image: Image, image path, or image tensor.
    :param entities: List of entity information returned by Kosmos-2 or OWL. This is expected to be a list of tuples (entity_name, None, list of bboxes)
    """
    if isinstance(image, Image.Image):
        image_h = image.height
        image_w = image.width
        image = np.array(image)[:, :, [2, 1, 0]]
    elif isinstance(image, str):
        if os.path.exists(image):
            pil_img = Image.open(image).convert("RGB")
            image = np.array(pil_img)[:, :, [2, 1, 0]]
            image_h = pil_img.height
            image_w = pil_img.width
        else:
            raise ValueError(f"invaild image path, {image}")
    elif isinstance(image, torch.Tensor):
        # pdb.set_trace()
        image_tensor = image.cpu()
        reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
        reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
        image_tensor = image_tensor * reverse_norm_std + reverse_norm_mean
        pil_img = T.ToPILImage()(image_tensor)
        image_h = pil_img.height
        image_w = pil_img.width
        image = np.array(pil_img)[:, :, [2, 1, 0]]
    else:
        raise ValueError(f"invaild image format, {type(image)} for {image}")

    if len(entities) == 0:
        return image

    new_image = image.copy()
    previous_bboxes = []
    # size of text
    text_size = 1
    # thickness of text
    text_line = 1  # int(max(1 * min(image_h, image_w) / 512, 1))
    box_line = 3
    (c_width, text_height), _ = cv2.getTextSize("F", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
    base_height = int(text_height * 0.675)
    text_offset_original = text_height - base_height
    text_spaces = 3

    for entity_name, _, bboxes in entities: # NOTE: "_" is returned by Kosmos-2 as a tuple (start, end); but it's never used
        for box in bboxes:
            if type(box) != BoundingBox:
                box = BoundingBox(*box)
            x1_norm, y1_norm, x2_norm, y2_norm = box.coords
            orig_x1, orig_y1, orig_x2, orig_y2 = int(x1_norm * image_w), int(y1_norm * image_h), int(x2_norm * image_w), int(y2_norm * image_h)
            # draw bbox
            # random color
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            new_image = cv2.rectangle(new_image, (orig_x1, orig_y1), (orig_x2, orig_y2), color, box_line)

            l_o, r_o = box_line // 2 + box_line % 2, box_line // 2 + box_line % 2 + 1

            x1 = orig_x1 - l_o
            y1 = orig_y1 - l_o

            if y1 < text_height + text_offset_original + 2 * text_spaces:
                y1 = orig_y1 + r_o + text_height + text_offset_original + 2 * text_spaces
                x1 = orig_x1 + r_o

            # add text background
            (text_width, text_height), _ = cv2.getTextSize(f"  {entity_name}", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
            text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = x1, y1 - (text_height + text_offset_original + 2 * text_spaces), x1 + text_width, y1

            for prev_bbox in previous_bboxes:
                while is_overlapping((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox):
                    text_bg_y1 += (text_height + text_offset_original + 2 * text_spaces)
                    text_bg_y2 += (text_height + text_offset_original + 2 * text_spaces)
                    y1 += (text_height + text_offset_original + 2 * text_spaces)

                    if text_bg_y2 >= image_h:
                        text_bg_y1 = max(0, image_h - (text_height + text_offset_original + 2 * text_spaces))
                        text_bg_y2 = image_h
                        y1 = image_h
                        break

            alpha = 0.5
            for i in range(text_bg_y1, text_bg_y2):
                for j in range(text_bg_x1, text_bg_x2+120):
                    if i < image_h and j < image_w:
                        if j < text_bg_x1 + 1.35 * c_width:
                            # original color
                            bg_color = color
                        else:
                            # white
                            bg_color = [255, 255, 255]
                        new_image[i, j] = (alpha * new_image[i, j] + (1 - alpha) * np.array(bg_color)).astype(np.uint8)

            cv2.putText(
                new_image, f"  {entity_name} ({round(box.score, 2)})", (x1, y1 - text_offset_original - 1 * text_spaces), cv2.FONT_HERSHEY_COMPLEX, text_size, (0, 0, 0), text_line, cv2.LINE_AA
            )
            # previous_locations.append((x1, y1))
            previous_bboxes.append((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2))

    pil_image = Image.fromarray(new_image[:, :, [2, 1, 0]])
    if save_path:
        pil_image.save(save_path)
    if show:
        plt.figure()
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.imshow(pil_image)
        plt.show()
    return new_image

