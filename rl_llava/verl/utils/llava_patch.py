import logging, inspect
log = logging.getLogger(__name__)

# 1) 给 LlavaConfig 加 max_position_embeddings（若缺失）
try:
    from transformers.models.llava.configuration_llava import LlavaConfig
    has_param = "max_position_embeddings" in str(inspect.signature(LlavaConfig.__init__))
    if not has_param:
        _orig_init = LlavaConfig.__init__
        def _patched_init(self, *args, max_position_embeddings=32768, **kwargs):
            _orig_init(self, *args, **kwargs)
            if not getattr(self, "max_position_embeddings", None):
                inherited = getattr(getattr(self, "text_config", None), "max_position_embeddings", None)
                self.max_position_embeddings = inherited if inherited is not None else max_position_embeddings
        LlavaConfig.__init__ = _patched_init
        cfg_status = "patched"
    else:
        cfg_status = "ok"
except Exception as e:
    cfg_status = f"error: {e}"

# 2) 修正 _no_split_modules 为类属性
try:
    import transformers.models.llava.modeling_llava as ml
    if hasattr(ml, "LlavaPreTrainedModel"):
        ml.LlavaPreTrainedModel._no_split_modules = ["CLIPEncoderLayer", "LlamaDecoderLayer"]
    else:
        ml._no_split_modules = ["CLIPEncoderLayer", "LlamaDecoderLayer"]
    split_status = "patched"
except Exception as e:
    split_status = f"error: {e}"

# 3) 打一行日志便于确认补丁在 worker 里生效
try:
    import transformers
    ver = getattr(transformers, "__version__", "?")
    path = getattr(transformers, "__file__", "?")
    log.warning("[llava_patch] transformers=%s ver=%s cfg=%s split=%s", path, ver, cfg_status, split_status)
except Exception:
    pass