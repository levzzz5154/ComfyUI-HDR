import os
import torch
import numpy as np
from PIL import Image, ImageOps
try:
    from PIL import ImageCms
    HAS_IMAGECMS = True
except ImportError:
    HAS_IMAGECMS = False
import folder_paths
import hashlib

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


def get_icc_profiles():
    profiles = set()
    dirs_to_scan = []
    try:
        input_dir = folder_paths.get_input_directory()
        if input_dir and os.path.exists(input_dir):
            dirs_to_scan.append(input_dir)
    except Exception:
        pass
    try:
        user_dir = folder_paths.get_user_directory()
        if user_dir and os.path.exists(user_dir):
            dirs_to_scan.append(user_dir)
    except Exception:
        pass

    for d in dirs_to_scan:
        try:
            for f in os.listdir(d):
                if f.lower().endswith(('.icc', '.icm')):
                    if os.path.isfile(os.path.join(d, f)):
                        profiles.add(f)
        except Exception:
            pass
    return sorted(list(profiles))


def get_icc_profile_combo():
    profiles = get_icc_profiles()
    combo = ["None"] + profiles
    default = "HDR.icc" if "HDR.icc" in profiles else "None"
    return combo, default


def resolve_icc_profile_path(profile_name):
    if not profile_name or profile_name == "None":
        return ""

    try:
        user_dir = folder_paths.get_user_directory()
        if user_dir:
            path = os.path.join(user_dir, profile_name)
            if os.path.exists(path):
                return path
    except Exception:
        pass

    try:
        input_dir = folder_paths.get_input_directory()
        if input_dir:
            path = os.path.join(input_dir, profile_name)
            if os.path.exists(path):
                return path
    except Exception:
        pass

    return ""


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
        profiles, default_profile = get_icc_profile_combo()
        return {
            "required": {
                "image": (sorted(hdr_files), {"image_upload": True}),
                "tonemap": ("BOOLEAN", {"default": True}),
                "gamma": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 4.0, "step": 0.1}),
            },
            "optional": {
                "use_camera_wb": ("BOOLEAN", {"default": False}),
                "icc_profile": (profiles, {"default": default_profile}),
            }
        }

    CATEGORY = "image/HDR"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask", "mask_inverted")
    FUNCTION = "load_image"

    def _apply_icc_profile(self, img, icc_profile_path):
        if not icc_profile_path or not os.path.exists(icc_profile_path):
            return img

        if not HAS_IMAGECMS:
            print("Warning: PIL.ImageCms is not available. Skipping ICC profile application.")
            return img

        try:
            input_profile = ImageCms.getProfile(icc_profile_path)
            output_profile = ImageCms.createProfile("sRGB")

            try:
                transformed = ImageCms.profileToProfile(img, input_profile, output_profile)
                if transformed is not None:
                    img = transformed
            except Exception as transform_error:
                print(f"Warning: Direct ICC transform failed on mode {img.mode}: {transform_error}. Attempting fallback conversion.")
                if img.mode in ['I;16', 'I;16B', 'I;16L', 'I', 'RGB;16', 'RGBA;16']:
                    img_conv = img.convert('RGB')
                    transformed = ImageCms.profileToProfile(img_conv, input_profile, output_profile)
                    if transformed is not None:
                        img = transformed
                else:
                    raise transform_error
        except Exception as e:
            print(f"Warning: Failed to apply ICC profile {icc_profile_path}: {e}")

        return img

    def _extract_mask(self, image_path, h, w):
        ext = os.path.splitext(image_path)[1].lower()
        if ext == '.exr':
            if HAS_CV2:
                img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                if img is not None and len(img.shape) == 3 and img.shape[2] == 4:
                    alpha = img[:, :, 3].astype(np.float32)
                    max_val = alpha.max()
                    if max_val > 1.0:
                        alpha = alpha / max_val
                    return 1.0 - torch.from_numpy(alpha)
        elif ext == '.dng':
            pass
        else:
            if HAS_CV2:
                img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if img is not None and len(img.shape) == 3 and img.shape[2] == 4:
                    alpha = img[:, :, 3].astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0)
                    return 1.0 - torch.from_numpy(alpha)
            else:
                try:
                    img = Image.open(image_path)
                    if 'A' in img.getbands():
                        chan = img.getchannel('A')
                        mask_np = np.array(chan).astype(np.float32) / (65535.0 if '16' in chan.mode else 255.0)
                        return 1.0 - torch.from_numpy(mask_np)
                    elif img.mode == 'P' and 'transparency' in img.info:
                        rgba = img.convert('RGBA')
                        mask_np = np.array(rgba.getchannel('A')).astype(np.float32) / 255.0
                        return 1.0 - torch.from_numpy(mask_np)
                except Exception:
                    pass

        # Always returns exact same dimensions as the loaded image
        return torch.zeros((h, w), dtype=torch.float32, device="cpu")

    def load_image(self, image, tonemap=True, gamma=2.2, use_camera_wb=False, icc_profile="None"):
        image_path = folder_paths.get_annotated_filepath(image)
        ext = os.path.splitext(image)[1].lower()

        icc_profile_path = resolve_icc_profile_path(icc_profile)

        if ext == '.dng':
            img_tuple = self._load_dng(image_path, tonemap, gamma, use_camera_wb, icc_profile_path)
        elif ext in ['.png', '.tiff', '.tif']:
            img_tuple = self._load_16bit_image(image_path, tonemap, gamma, icc_profile_path)
        elif ext == '.exr':
            img_tuple = self._load_exr(image_path, tonemap, gamma, icc_profile_path)
        else:
            img_tuple = self._load_standard_image(image_path, tonemap, gamma, icc_profile_path)

        img = img_tuple[0]
        h, w = img.shape[1], img.shape[2]
        mask = self._extract_mask(image_path, h, w)

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        mask_inverted = 1.0 - mask
        return (img, mask, mask_inverted)

    def _load_dng(self, image_path, tonemap, gamma, use_camera_wb=False, icc_profile_path=""):
        if not HAS_RAWPY:
            raise ImportError("rawpy is required to load DNG files. Install with: pip install rawpy")

        has_icc = bool(icc_profile_path and os.path.exists(icc_profile_path))
        if has_icc:
            print("Warning: ICC profile application is not supported for DNG format (RawPy).")

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

        if tonemap and not has_icc:
            image = apply_gammaTonemap(image, gamma)

        return (image,)

    def _load_16bit_image(self, image_path, tonemap, gamma, icc_profile_path=""):
        has_icc = icc_profile_path and os.path.exists(icc_profile_path)
        if HAS_CV2 and not has_icc:
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

            if has_icc:
                img = self._apply_icc_profile(img, icc_profile_path)

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

            if has_icc:
                image = apply_inverse_gammaTonemap(image, 2.2)

        if tonemap and not has_icc:
            image = apply_gammaTonemap(image, gamma)

        return (image,)

    def _load_exr(self, image_path, tonemap, gamma, icc_profile_path=""):
        has_icc = bool(icc_profile_path and os.path.exists(icc_profile_path))
        if has_icc:
            print("Warning: ICC profile application is not supported for EXR format (OpenCV).")
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

        if tonemap and not has_icc:
            image = apply_gammaTonemap(image, gamma)

        return (image,)

    def _load_standard_image(self, image_path, tonemap, gamma, icc_profile_path=""):
        has_icc = icc_profile_path and os.path.exists(icc_profile_path)
        if HAS_CV2 and not has_icc:
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

            if has_icc:
                img = self._apply_icc_profile(img, icc_profile_path)

            if img.mode == 'I':
                img = img.point(lambda i: i * (1 / 255))

            arr = np.array(img.convert('RGB'), dtype=np.float32) / 255.0
            image = torch.from_numpy(arr)[None,]

            if has_icc:
                image = apply_inverse_gammaTonemap(image, 2.2)

        if tonemap and not has_icc:
            image = apply_gammaTonemap(image, gamma)

        return (image,)

    @classmethod
    def IS_CHANGED(s, image, tonemap=True, gamma=2.2, icc_profile="None"):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hash(image_path + str(tonemap) + str(gamma) + str(icc_profile))
        return m

    @classmethod
    def VALIDATE_INPUTS(s, image, tonemap=True, gamma=2.2, icc_profile="None"):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        if icc_profile and icc_profile != "None":
            resolved = resolve_icc_profile_path(icc_profile)
            if not resolved or not os.path.isfile(resolved):
                return "Invalid ICC profile: {}".format(icc_profile)
        return True


class LoadHDRImagePath:
    @staticmethod
    def _normalize_path(path: str) -> str:
        if not path:
            return path
        path = path.strip()
        path = path.replace('\\', '/')
        path = os.path.normpath(path)
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        return path

    @classmethod
    def INPUT_TYPES(s):
        profiles, default_profile = get_icc_profile_combo()
        return {
            "required": {
                "image_path": ("STRING", {"default": "", "multiline": False}),
                "tonemap": ("BOOLEAN", {"default": True}),
                "gamma": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 4.0, "step": 0.1}),
            },
            "optional": {
                "use_camera_wb": ("BOOLEAN", {"default": False}),
                "icc_profile": (profiles, {"default": default_profile}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask", "mask_inverted")
    FUNCTION = "load_image"
    CATEGORY = "image/HDR"

    def load_image(self, image_path, tonemap=True, gamma=2.2, use_camera_wb=False, icc_profile="None"):
        normalized_path = self._normalize_path(image_path)
        if not normalized_path or not os.path.isfile(normalized_path):
            raise ValueError(f"Invalid image path: {image_path} (resolved to: {normalized_path})")

        ext = os.path.splitext(normalized_path)[1].lower()

        icc_profile_path = resolve_icc_profile_path(icc_profile)

        loader = LoadHDRImage()
        if ext == '.dng':
            img_tuple = loader._load_dng(normalized_path, tonemap, gamma, use_camera_wb, icc_profile_path)
        elif ext in ['.png', '.tiff', '.tif']:
            img_tuple = loader._load_16bit_image(normalized_path, tonemap, gamma, icc_profile_path)
        elif ext == '.exr':
            img_tuple = loader._load_exr(normalized_path, tonemap, gamma, icc_profile_path)
        else:
            img_tuple = loader._load_standard_image(normalized_path, tonemap, gamma, icc_profile_path)

        img = img_tuple[0]
        h, w = img.shape[1], img.shape[2]
        mask = loader._extract_mask(normalized_path, h, w)

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        mask_inverted = 1.0 - mask
        return (img, mask, mask_inverted)

    @classmethod
    def IS_CHANGED(s, image_path, tonemap=True, gamma=2.2, use_camera_wb=False, icc_profile="None"):
        normalized_path = s._normalize_path(image_path)
        if not normalized_path or not os.path.isfile(normalized_path):
            return ""
        m = hashlib.sha256()
        try:
            with open(normalized_path, 'rb') as f:
                m.update(f.read())
            m.update(str(tonemap).encode('utf-8'))
            m.update(str(gamma).encode('utf-8'))
            m.update(str(use_camera_wb).encode('utf-8'))
            m.update(str(icc_profile).encode('utf-8'))
        except Exception:
            pass
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image_path, tonemap=True, gamma=2.2, use_camera_wb=False, icc_profile="None"):
        if not image_path:
            return "Image path cannot be empty"
        normalized_path = s._normalize_path(image_path)
        if not os.path.isfile(normalized_path):
            return f"Invalid image file: {image_path} (resolved to: {normalized_path})"
        if icc_profile and icc_profile != "None":
            resolved = resolve_icc_profile_path(icc_profile)
            if not resolved or not os.path.isfile(resolved):
                return f"Invalid ICC profile: {icc_profile}"
        return True


class LoadHDRImageDirectory:
    @staticmethod
    def _normalize_path(path: str) -> str:
        if not path:
            return path
        path = path.strip()
        path = path.replace('\\', '/')
        path = os.path.normpath(path)
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        return path

    @classmethod
    def INPUT_TYPES(s):
        profiles, default_profile = get_icc_profile_combo()
        return {
            "required": {
                "directory_path": ("STRING", {"default": "", "multiline": False}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "load_count": ("INT", {"default": 1, "min": 1, "max": 1024, "step": 1}),
                "tonemap": ("BOOLEAN", {"default": True}),
                "gamma": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 4.0, "step": 0.1}),
            },
            "optional": {
                "use_camera_wb": ("BOOLEAN", {"default": False}),
                "icc_profile": (profiles, {"default": default_profile}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("images", "masks", "masks_inverted")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "load_images"
    CATEGORY = "image/HDR"

    def load_images(self, directory_path, start_index, load_count, tonemap=True, gamma=2.2, use_camera_wb=False, icc_profile="None"):
        normalized_path = self._normalize_path(directory_path)

        if not normalized_path or not os.path.isdir(normalized_path):
            raise ValueError(f"Invalid directory path: {directory_path}")

        valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif', '.dng', '.exr'}
        files = []
        for f in os.listdir(normalized_path):
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_extensions:
                files.append(os.path.join(normalized_path, f))

        files.sort()

        end_index = start_index + load_count
        selected_files = files[start_index:end_index]

        if not selected_files:
             raise ValueError(f"No HDR/DNG/EXR/Standard images found in range [{start_index}:{end_index}] in directory: {directory_path}")

        output_images = []
        output_masks = []
        output_masks_inverted = []

        icc_profile_path = resolve_icc_profile_path(icc_profile)

        loader = LoadHDRImage()

        for file_path in selected_files:
            try:
                ext = os.path.splitext(file_path)[1].lower()
                if ext == '.dng':
                    img_tuple = loader._load_dng(file_path, tonemap, gamma, use_camera_wb, icc_profile_path)
                elif ext in ['.png', '.tiff', '.tif']:
                    img_tuple = loader._load_16bit_image(file_path, tonemap, gamma, icc_profile_path)
                elif ext == '.exr':
                    img_tuple = loader._load_exr(file_path, tonemap, gamma, icc_profile_path)
                else:
                    img_tuple = loader._load_standard_image(file_path, tonemap, gamma, icc_profile_path)
            except Exception as e:
                print(f"Warning: Could not load HDR image {file_path}: {e}")
                continue

            img = img_tuple[0]
            h, w = img.shape[1], img.shape[2]
            mask = loader._extract_mask(file_path, h, w)

            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)

            mask_inverted = 1.0 - mask

            output_images.append(img)
            output_masks.append(mask)
            output_masks_inverted.append(mask_inverted)

        if not output_images:
            raise ValueError("No valid HDR/DNG/EXR/Standard images loaded.")

        return (output_images, output_masks, output_masks_inverted)

    @classmethod
    def IS_CHANGED(s, directory_path, start_index, load_count, tonemap=True, gamma=2.2, use_camera_wb=False, icc_profile="None"):
        normalized_path = s._normalize_path(directory_path)
        if not normalized_path or not os.path.isdir(normalized_path):
            return ""

        valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif', '.dng', '.exr'}
        files = []
        try:
            for f in os.listdir(normalized_path):
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_extensions:
                    files.append(os.path.join(normalized_path, f))
        except Exception:
            return float("NaN")

        files.sort()
        end_index = start_index + load_count
        selected_files = files[start_index:end_index]

        m = hashlib.sha256()
        for p in selected_files:
            try:
                m.update(p.encode('utf-8'))
                m.update(str(os.path.getmtime(p)).encode('utf-8'))
            except Exception:
                pass
        m.update(str(tonemap).encode('utf-8'))
        m.update(str(gamma).encode('utf-8'))
        m.update(str(use_camera_wb).encode('utf-8'))
        m.update(str(icc_profile).encode('utf-8'))
        return m.digest().hex()


class HDRVAEEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
                "dtype": (["auto", "fp32"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "latent/HDR"

    def encode(self, vae, pixels, dtype="auto"):
        if dtype == "fp32":
            original_dtype = next(vae.first_stage_model.parameters()).dtype
            vae.first_stage_model = vae.first_stage_model.float()
            t = vae.encode(pixels)
            vae.first_stage_model = vae.first_stage_model.to(original_dtype)
        else:
            t = vae.encode(pixels)
        return ({"samples": t},)


class HDRVAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "dtype": (["auto", "fp32"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "latent/HDR"

    def decode(self, vae, samples, dtype="auto"):
        latent = samples["samples"]
        if hasattr(latent, 'is_nested') and latent.is_nested:
            latent = latent.unbind()[0]

        if dtype == "fp32":
            original_dtype = next(vae.first_stage_model.parameters()).dtype
            vae.first_stage_model = vae.first_stage_model.float()
            images = vae.decode(latent)
            vae.first_stage_model = vae.first_stage_model.to(original_dtype)
        else:
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
        profiles, default_profile = get_icc_profile_combo()
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "HDR/ComfyUI"}),
                "tonemap_for_viewing": ("BOOLEAN", {"default": False}),
                "gamma": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 4.0, "step": 0.1}),
                "icc_profile": (profiles, {"default": default_profile}),
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

    def save_images(self, images, filename_prefix="HDR/ComfyUI", tonemap_for_viewing=False, gamma=2.2, icc_profile="None", prompt=None, extra_pnginfo=None):
        import json
        import PIL.PngImagePlugin
        try:
            from comfy.cli_args import args
            disable_metadata = args.disable_metadata
        except Exception:
            disable_metadata = False

        metadata = None
        if not disable_metadata:
            metadata = PIL.PngImagePlugin.PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for k, v in extra_pnginfo.items():
                    metadata.add_text(k, json.dumps(v))

        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
        )

        resolved_path = resolve_icc_profile_path(icc_profile) if (icc_profile and icc_profile != "None") else ""
        has_icc = bool(resolved_path and os.path.exists(resolved_path))
        profile_bytes = None
        if has_icc:
            try:
                with open(resolved_path, 'rb') as f:
                    profile_bytes = f.read()
            except Exception as e:
                print(f"Warning: Failed to read ICC profile {resolved_path}: {e}")

        results = []
        for batch_number, image in enumerate(images):
            if self.type == "temp" and has_icc:
                save_image = apply_gammaTonemap(image, 2.2)
            elif tonemap_for_viewing:
                save_image = apply_gammaTonemap(image, gamma)
            else:
                save_image = image

            save_image = save_image.clamp(0, 1)

            applied_preview_transform = False
            img_pil_transformed = None

            if self.type == "temp" and has_icc and HAS_IMAGECMS:
                try:
                    img_pil_src = Image.fromarray((save_image.cpu().numpy() * 255.0).astype(np.uint8), mode='RGB')
                    input_profile = ImageCms.createProfile("sRGB")
                    output_profile = ImageCms.getProfile(resolved_path)
                    transformed = ImageCms.profileToProfile(img_pil_src, input_profile, output_profile)
                    if transformed is not None:
                        img_pil_transformed = transformed
                        img_array = (np.array(transformed).astype(np.uint32) * 257).astype(np.uint16)
                        applied_preview_transform = True
                except Exception as e:
                    print(f"Warning: Failed to apply display transform: {e}")

            if not applied_preview_transform:
                img_array = (save_image.cpu().numpy() * 65535.0).astype(np.uint16)

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            filepath = os.path.join(full_output_folder, file)

            if HAS_CV2:
                cv2.imwrite(filepath, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                if metadata is not None or (self.type == "output" and profile_bytes is not None):
                    try:
                        import struct
                        import zlib
                        with open(filepath, 'rb') as f:
                            data = f.read()
                        if data[:8] == b'\x89PNG\r\n\x1a\n':
                            ihdr_len = struct.unpack('>I', data[8:12])[0]
                            ihdr_end = 8 + 4 + 4 + ihdr_len + 4
                            new_chunks = []
                            if self.type == "output" and profile_bytes is not None:
                                chunk_data = b'icc\x00\x00' + zlib.compress(profile_bytes)
                                chunk_type = b'iCCP'
                                crc = zlib.crc32(chunk_type + chunk_data) & 0xffffffff
                                chunk = struct.pack('>I', len(chunk_data)) + chunk_type + chunk_data + struct.pack('>I', crc)
                                new_chunks.append(chunk)

                            if metadata is not None:
                                for chunk_type, chunk_data, after_idat in metadata.chunks:
                                    crc = zlib.crc32(chunk_type + chunk_data) & 0xffffffff
                                    chunk = struct.pack('>I', len(chunk_data)) + chunk_type + chunk_data + struct.pack('>I', crc)
                                    new_chunks.append(chunk)
                            new_data = data[:ihdr_end] + b''.join(new_chunks) + data[ihdr_end:]
                            with open(filepath, 'wb') as f:
                                f.write(new_data)
                    except Exception as e:
                        print(f"Warning: Failed to write metadata to 16-bit PNG: {e}")
            else:
                if applied_preview_transform and img_pil_transformed is not None:
                    img_pil_transformed.save(filepath, pnginfo=metadata)
                else:
                    img_array_8bit = (img_array / 256).astype(np.uint8)
                    img_pil = Image.fromarray(img_array_8bit, mode='RGB')
                    if self.type == "output" and profile_bytes is not None:
                        img_pil.save(filepath, pnginfo=metadata, icc_profile=profile_bytes)
                    else:
                        img_pil.save(filepath, pnginfo=metadata)

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
        profiles, default_profile = get_icc_profile_combo()
        return {
            "required": {
                "images": ("IMAGE",),
                "tonemap_for_viewing": ("BOOLEAN", {"default": True}),
                "gamma": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 4.0, "step": 0.1}),
                "icc_profile": (profiles, {"default": default_profile}),
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


class LinearToSRGB:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "gamma": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 4.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    CATEGORY = "image/HDR"

    def convert(self, image, gamma=2.2):
        return (torch.pow(image.clamp(0, 1), 1.0 / gamma),)


class SRGBToLinear:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "gamma": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 4.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    CATEGORY = "image/HDR"

    def convert(self, image, gamma=2.2):
        return (torch.pow(image.clamp(0, 1), gamma),)


class SaveLatentToNpy:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "filename_prefix": ("STRING", {"default": "latent"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "latent/HDR"

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    def save(self, samples, filename_prefix="latent"):
        latent = samples["samples"]

        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, latent.shape[2], latent.shape[1]
        )

        filepath = os.path.join(full_output_folder, f"{filename}_{counter:05}_.npy")
        np.save(filepath, latent.cpu().numpy())

        results = [{
            "filename": f"{filename}_{counter:05}_.npy",
            "subfolder": subfolder,
            "type": "output"
        }]

        return {"ui": {"files": results}}
