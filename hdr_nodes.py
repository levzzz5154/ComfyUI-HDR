import os
import torch
import numpy as np
from PIL import Image, ImageOps
import folder_paths

try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def apply_gammaTonemap(linear, gamma=2.2):
    return torch.pow(linear.clamp(0, 1), 1.0 / gamma)


def apply_inverse_gammaTonemap(srgb, gamma=2.2):
    return torch.pow(srgb.clamp(0, 1), gamma)


class LoadHDRImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        hdr_files = []
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in ['.png', '.dng', '.tiff', '.tif', '.exr']:
                hdr_files.append(f)
        return {
            "required": {
                "image": (sorted(hdr_files), {"image_upload": True}),
                "tonemap": ("BOOLEAN", {"default": True}),
                "gamma": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 4.0, "step": 0.1}),
            },
            "optional": {
                "use_camera_wb": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "image/HDR"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"

    def load_image(self, image, tonemap=True, gamma=2.2, use_camera_wb=False):
        image_path = folder_paths.get_annotated_filepath(image)
        ext = os.path.splitext(image)[1].lower()

        if ext == '.dng':
            return self._load_dng(image_path, tonemap, gamma, use_camera_wb)
        elif ext in ['.png', '.tiff', '.tif']:
            return self._load_16bit_image(image_path, tonemap, gamma)
        elif ext == '.exr':
            return self._load_exr(image_path, tonemap, gamma)
        else:
            return self._load_standard_image(image_path, tonemap, gamma)

    def _load_dng(self, image_path, tonemap, gamma, use_camera_wb=False):
        if not HAS_RAWPY:
            raise ImportError("rawpy is required to load DNG files. Install with: pip install rawpy")

        with rawpy.imread(image_path) as raw:
            rgb = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                output_bps=16,
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                use_camera_wb=use_camera_wb,
            )

        image = rgb.astype(np.float32) / 65535.0
        image = torch.from_numpy(image)[None,]

        if tonemap:
            image = apply_gammaTonemap(image, gamma)

        return (image,)

    def _load_16bit_image(self, image_path, tonemap, gamma):
        if HAS_CV2:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if img.dtype == np.uint16:
                image = img.astype(np.float32) / 65535.0
            elif img.dtype == np.uint8:
                image = img.astype(np.float32) / 255.0
            else:
                image = img.astype(np.float32)
            
            image = torch.from_numpy(image)[None,]
        else:
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)
            
            if img.mode == 'I;16':
                arr = np.array(img, dtype=np.float32) / 65535.0
                image = torch.from_numpy(arr)[None, ..., None]
                image = image.expand(-1, -1, -1, 3)
            elif img.mode == 'I;16B':
                arr = np.array(img, dtype=np.float32) / 65535.0
                image = torch.from_numpy(arr)[None, ..., None]
                image = image.expand(-1, -1, -1, 3)
            elif img.mode == 'I':
                arr = np.array(img, dtype=np.float32) / 65535.0
                image = torch.from_numpy(arr)[None, ..., None]
                image = image.expand(-1, -1, -1, 3)
            elif img.mode == 'RGB;16' or img.mode == 'RGBA;16':
                arr = np.array(img, dtype=np.float32) / 65535.0
                if arr.shape[-1] == 4:
                    arr = arr[..., :3]
                image = torch.from_numpy(arr)[None,]
            else:
                arr = np.array(img.convert('RGB'), dtype=np.float32) / 255.0
                image = torch.from_numpy(arr)[None,]

        if tonemap:
            image = apply_gammaTonemap(image, gamma)

        return (image,)

    def _load_exr(self, image_path, tonemap, gamma):
        if HAS_CV2:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                raise ValueError(f"Could not load EXR: {image_path}")
            
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            image = img.astype(np.float32)
            max_val = image.max()
            if max_val > 1.0:
                image = image / max_val
            
            image = torch.from_numpy(image)[None,]
        else:
            raise ImportError("cv2 is required to load EXR files. Install with: pip install opencv-python")

        if tonemap:
            image = apply_gammaTonemap(image, gamma)

        return (image,)

    def _load_standard_image(self, image_path, tonemap, gamma):
        if HAS_CV2:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if img.dtype == np.uint16:
                image = img.astype(np.float32) / 65535.0
            else:
                image = img.astype(np.float32) / 255.0
            
            image = torch.from_numpy(image)[None,]
        else:
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)

            if img.mode == 'I':
                img = img.point(lambda i: i * (1 / 255))

            arr = np.array(img.convert('RGB'), dtype=np.float32) / 255.0
            image = torch.from_numpy(arr)[None,]

        if tonemap:
            image = apply_gammaTonemap(image, gamma)

        return (image,)

    @classmethod
    def IS_CHANGED(s, image, tonemap=True, gamma=2.2):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hash(image_path + str(tonemap) + str(gamma))
        return m

    @classmethod
    def VALIDATE_INPUTS(s, image, tonemap=True, gamma=2.2):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True


class HDRVAEEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "latent/HDR"

    def encode(self, vae, pixels):
        t = vae.encode(pixels)
        return ({"samples": t},)


class HDRVAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "latent/HDR"

    def decode(self, vae, samples):
        latent = samples["samples"]
        if hasattr(latent, 'is_nested') and latent.is_nested:
            latent = latent.unbind()[0]

        images = vae.decode(latent)
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images,)


class HDRSaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "HDR/ComfyUI"}),
                "tonemap_for_viewing": ("BOOLEAN", {"default": False}),
                "gamma": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 4.0, "step": 0.1}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image/HDR"

    def save_images(self, images, filename_prefix="HDR/ComfyUI", tonemap_for_viewing=False, gamma=2.2, prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
        )

        results = []
        for batch_number, image in enumerate(images):
            if tonemap_for_viewing:
                save_image = apply_gammaTonemap(image, gamma)
            else:
                save_image = image

            save_image = save_image.clamp(0, 1)
            img_array = (save_image.cpu().numpy() * 65535.0).astype(np.uint16)

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            filepath = os.path.join(full_output_folder, file)

            if HAS_CV2:
                cv2.imwrite(filepath, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            else:
                img_array_8bit = (img_array / 256).astype(np.uint8)
                img_pil = Image.fromarray(img_array_8bit, mode='RGB')
                img_pil.save(filepath)

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"ui": {"images": results}}


class HDRPreviewImage(HDRSaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(str(np.random.randint(0, 26)) for _ in range(5))

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "tonemap_for_viewing": ("BOOLEAN", {"default": True}),
                "gamma": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 4.0, "step": 0.1}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }


class TonemapImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["linear_to_srgb", "srgb_to_linear", "reinhard", "aces"], {"default": "linear_to_srgb"}),
                "gamma": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 4.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "tonemap"

    CATEGORY = "image/HDR"

    def tonemap(self, image, mode, gamma=2.2):
        if mode == "linear_to_srgb":
            result = apply_gammaTonemap(image, gamma)
        elif mode == "srgb_to_linear":
            result = apply_inverse_gammaTonemap(image, gamma)
        elif mode == "reinhard":
            result = image / (image + 1.0)
        elif mode == "aces":
            a = 2.51
            b = 0.03
            c = 2.43
            d = 0.59
            e = 0.14
            result = (image * (a * image + b)) / (image * (c * image + d) + e)
        else:
            result = image

        return (result.clamp(0, 1),)
