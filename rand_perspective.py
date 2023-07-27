import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms
from torch import Tensor
from PIL import Image


def random_perspective_about_center(img, distortion_scale = 0.5, fill=100):
    """Takes in input image and returns image with random perspective transformation applied, 
    preserving center of image"""
    height, width = 128, 128
    half_height = height // 2
    half_width = width // 2
    topleft = [
        int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
        int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()),
    ]
    topright = [
        int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
        int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()),
    ]
    botright = [
        int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
        int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item()),
    ]
    botleft = [
        int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
        int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item()),
    ]
    startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    endpoints = [topleft, topright, botright, botleft]
    center = line_intersection([topleft, botright], [topright, botleft])
    print(center)
    adjust = [half_width - round(center[0]), half_height - round(center[1])]
    print(adjust)

    # Apply perspective transformation using torchvision.transforms.functional.perspective
    perspective_img = F.perspective(img, startpoints, endpoints, fill=fill)
    centered_img = F.affine(perspective_img, angle=0, translate=adjust, scale=1, shear=0)
    
    # Zooms in slightly to preserve original image size
    cropped_img = F.center_crop(centered_img, [100, 100])
    resized_img = F.resize(cropped_img, [128, 128])

    return resized_img


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]


# Example usage:
input_image = Image.open("path_to_image")
output_image = random_perspective_about_center(input_image)
output_image.show()

class RandomPerspectiveAboutCenter(torch.nn.Module):
    """Performs a random perspective transformation of the given image with a given probability, preserving the center of the image.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.
            Default is 0.5.
        p (float): probability of the image being transformed. Default is 0.5.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        fill (sequence or number): Pixel fill value for the area outside the transformed
            image. Default is ``0``. If given a number, the value is used for all bands respectively.
    """

    def __init__(self, distortion_scale=0.5, p=0.5, interpolation=InterpolationMode.BILINEAR, fill=0):
        super().__init__()
        self.p = p

        self.interpolation = interpolation
        self.distortion_scale = distortion_scale

        if fill is None:
            fill = 0

        self.fill = fill

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be Perspectively transformed.

        Returns:
            PIL Image or Tensor: Randomly transformed image.
        """

        fill = self.fill
        channels, height, width = transforms.functional.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]

        if torch.rand(1) < self.p:
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
            center = line_intersection([endpoints[0], endpoints[2]], [endpoints[1], endpoints[3]])
            adjust = [width // 2 - round(center[0]), height // 2 - round(center[1])]
            perspective_img = transforms.functional.perspective(img, startpoints, endpoints, self.interpolation, fill)
            centered_img = transforms.functional.affine(perspective_img, angle=0, translate=adjust, scale=1, shear=0)
            cropped_img = transforms.functional.center_crop(centered_img, [100, 100])
            resized_img = transforms.functional.resize(cropped_img, [128, 128])
            return resized_img
        return img

    @staticmethod
    def get_params(width: int, height: int, distortion_scale: float):
        """Get parameters for ``perspective`` for a random perspective transform.

        Args:
            width (int): width of the image.
            height (int): height of the image.
            distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.

        Returns:
            List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
            List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        """
        half_height = height // 2
        half_width = width // 2
        topleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()),
        ]
        topright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()),
        ]
        botright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item()),
        ]
        botleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item()),
        ]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"
    
# Example usage:
input_image = Image.open("path_to_image")
distort = transforms.Compose([
    RandomPerspectiveAboutCenter()
])
output_image = distort(input_image)
output_image.show()

