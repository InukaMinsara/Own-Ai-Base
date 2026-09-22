from pathlib import Path
import io


class LocalVision:
    """Optional local image understanding.

    When transformers is installed, BLIP can produce a local caption.
    Without it, the analyzer still returns deterministic image metadata.
    No external vision API is required.
    """

    def __init__(self):
        self.processor = None
        self.model = None
        self.device = "cpu"

    def _load_blip(self):
        if self.model is not None:
            return

        try:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Local vision requires transformers and Pillow."
            ) from exc

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        self.model = (
            BlipForConditionalGeneration
            .from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            .to(self.device)
        )

        self.model.eval()

    def metadata(self, image_bytes):
        from PIL import Image, ImageStat

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        stat = ImageStat.Stat(image)

        return {
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "mean_rgb": [
                round(value, 2)
                for value in stat.mean
            ],
        }

    def describe(self, image_bytes, prompt=None):
        self._load_blip()

        import torch
        from PIL import Image

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        inputs = self.processor(
            images=image,
            text=prompt or None,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=80,
            )

        caption = self.processor.decode(
            output[0],
            skip_special_tokens=True,
        ).strip()

        return caption
