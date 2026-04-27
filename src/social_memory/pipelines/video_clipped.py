

from functools import partial

from social_memory.pipelines.video_baseline import VideoPipeline

from social_memory.transforms.video import (
    clip_around_oracle, encode_video, load_video
)

class VideoClippedPipeline(VideoPipeline):
    """
    
    """
    NAME = "video_clipped"

    def __init__(self, configs):
        super().__init__(configs)
        
        self.transforms = [
            load_video,
            partial(
                clip_around_oracle,
                **self.configs.transform_configs.get("clip_around_oracle", {})
            ),
            encode_video,
        ]