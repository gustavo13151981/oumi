from oumi.inference.remote.batch_mixin import AsyncBatchMixin
from oumi.inference.remote_inference_engine import RemoteInferenceEngine


class OpenAIInferenceEngine(RemoteInferenceEngine, AsyncBatchMixin):
    """Engine for running inference against a server implementing the OpenAI API."""
