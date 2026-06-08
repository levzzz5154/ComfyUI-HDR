from .hdr_nodes import LoadHDRImage, LoadHDRImagePath, LoadHDRImageDirectory, HDRVAEEncode, HDRVAEDecode, HDRSaveImage, HDRPreviewImage, TonemapImage, LinearToSRGB, SRGBToLinear, SaveLatentToNpy

NODE_CLASS_MAPPINGS = {
    "LoadHDRImage": LoadHDRImage,
    "LoadHDRImagePath": LoadHDRImagePath,
    "LoadHDRImageDirectory": LoadHDRImageDirectory,
    "HDRVAEEncode": HDRVAEEncode,
    "HDRVAEDecode": HDRVAEDecode,
    "HDRSaveImage": HDRSaveImage,
    "HDRPreviewImage": HDRPreviewImage,
    "TonemapImage": TonemapImage,
    "LinearToSRGB": LinearToSRGB,
    "SRGBToLinear": SRGBToLinear,
    "SaveLatentToNpy": SaveLatentToNpy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadHDRImage": "Load HDR Image",
    "LoadHDRImagePath": "Load HDR Image (Path)",
    "LoadHDRImageDirectory": "Load HDR Images (Directory)",
    "HDRVAEEncode": "HDR VAE Encode",
    "HDRVAEDecode": "HDR VAE Decode",
    "HDRSaveImage": "HDR Save Image",
    "HDRPreviewImage": "HDR Preview Image",
    "TonemapImage": "Tonemap Image",
    "LinearToSRGB": "Linear to sRGB",
    "SRGBToLinear": "sRGB to Linear",
    "SaveLatentToNpy": "Save Latent to NPY",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
