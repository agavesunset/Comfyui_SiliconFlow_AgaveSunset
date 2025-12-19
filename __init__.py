from .nodes import SiliconFlowLoader_AS, SiliconFlowSampler_AS

NODE_CLASS_MAPPINGS = {
    "SiliconFlowLoader_AS": SiliconFlowLoader_AS,
    "SiliconFlowSampler_AS": SiliconFlowSampler_AS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SiliconFlowLoader_AS": "SiliconFlowLoader_AS",
    "SiliconFlowSampler_AS": "SiliconFlowSampler_AS",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
