import torch
from cog import BasePredictor, Input, Path
from musetalk.utils.utils import load_all_model
import os
import numpy as np
from app import inference, check_video

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        # Load models just like in app.py
        self.audio_processor, self.vae, self.unet, self.pe = load_all_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timesteps = torch.tensor([0], device=self.device)

    def predict(
        self,
        audio: Path = Input(description="Input audio file"),
        video: Path = Input(description="Reference video file"), 
        bbox_shift: float = Input(description="BBox shift value in pixels", default=0)
    ) -> Path:
        """Run lip sync prediction"""
        
        # Process video to ensure 25fps
        processed_video = check_video(str(video))
        
        # Run inference
        output_path, _ = inference(
            str(audio),
            processed_video,
            bbox_shift
        )
        
        return Path(output_path)
