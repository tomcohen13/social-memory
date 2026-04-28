

from social_memory.pipelines.video_baseline import VideoPipeline
from social_memory.transforms import Transform
from social_memory.transforms.video import clip_around_oracle

class VideoClippedPipeline(VideoPipeline):
    """
    Video pipeline with clipping applied each video, applied around oracle.
    """
    NAME = "video_clipped"

    def __init__(self, configs):
        super().__init__(configs)
        clip_transform = Transform(
            clip_around_oracle,
            **configs.transform_configs.get("clip_around_oracle", {})
        )
        self.transforms.add_before(clip_transform, before_transform="encode_video")
