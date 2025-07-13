import torch
import logging
from typing import List, Dict, Tuple

from scheduler.env_encoder import ArmContextGenerator
from config import device_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Scheduler:
    def __init__(self, model_path: str, sr_model_features: List):
        try:
            context_generator = ArmContextGenerator(sr_model_features)
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            context_generator.load_state_dict(state_dict)
            self.context_generator = context_generator.to(device_name)
            logger.info("Scheduler has been successfully initialized")
        except Exception as e:
            logger.error(f"Scheduler initialization failed: {str(e)}", exc_info=True)
            self.model = None

        self.sr_model_features = sr_model_features

    def __call__(self, env: Dict[str, float], video_clip: torch.Tensor) -> Tuple[int, int]:
        bandwidth = env['bandwidth']
        sr_queue = env['sr_queue']
        buffer = env['buffer']
        sr_size = env['sr_size']

        # 客户端处理不考虑编码时间
        context = []
        for idx, model_feat in enumerate(self.model_feat):
            arm_feat = list(model_feat.values())
            arm_feat[0] *= env_array[-1] # k * size
            context.append(np.array(env_array[:-1] + arm_feat))
        context = np.array(context)
        self.cached_context = context
        self.cached_action = self.neural_ucb.take_action(context)
        return self.cached_action