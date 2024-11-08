from __future__ import annotations

import math

import torch

from oumi.builders import build_tokenizer
from oumi.core.configs import InferenceConfig, ModelParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.logging import logger

try:
    import sglang as sgl  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError:
    vllm = None


class SGLangInferenceEngine(BaseInferenceEngine):
    """Engine for running vllm inference locally."""

    def __init__(
        self,
        model_params: ModelParams,
        tensor_parallel_size: int = -1,
        gpu_memory_utilization: float = 1.0,
    ):
        """Initializes the SGLang inference Engine.

        Args:
            model_params: The model parameters to use for inference.
            tensor_parallel_size: The number of tensor parallel processes to use.
                If set to -1, we will use all the available GPUs.
            gpu_memory_utilization: The fraction of available GPU memory the model's
                executor will use. It can range from 0 to 1. Defaults to 1.0, i.e.,
                full (100%) memory utilization.
        """
        if not sgl:
            raise RuntimeError("SGLang (sgl) is not installed.")

        if (
            math.isfinite(gpu_memory_utilization)
            and gpu_memory_utilization > 0
            and gpu_memory_utilization <= 1.0
        ):
            raise ValueError(
                "GPU memory utilization must be within (0, 1]. "
                f"Actual: {gpu_memory_utilization}."
            )

        if tensor_parallel_size <= 0:
            if torch.cuda.device_count() > 1:
                tensor_parallel_size = torch.cuda.device_count()
            else:
                tensor_parallel_size = 1

        sgl_kwargs = {}
        self._lora_request = None
        if model_params.adapter_model:
            raise NotImplementedError("Adapter support is not implemented yet!")
        self._tokenizer = build_tokenizer(model_params)
        self._model_params = model_params
        self._sgl_runtime = sgl.Runtime(
            model=model_params.model_name,
            trust_remote_code=model_params.trust_remote_code,
            dtype=model_params.torch_dtype_str,
            mem_fraction_static=gpu_memory_utilization,
            tp_size=tensor_parallel_size,
            context_len=model_params.model_max_length,
            # port=?
            **sgl_kwargs,
        )

    def _convert_conversation_to_sgl_pipeline_impl(self, conversation: Conversation):
        for message in conversation.messages:
            pass

    @sgl.function
    def _convert_conversation_to_sgl_pipeline(self, conversation: Conversation):
        """Converts a conversation to a list of vllm input messages.

        Args:
            conversation: The conversation to convert.

        Returns:
            List[ChatCompletionMessageParam]: A list of vllm input messages.
        """
        return [
            {
                "content": message.content or "",
                "role": message.role,
            }
            for message in conversation.messages
        ]

    def _infer(
        self, input: list[Conversation], inference_config: InferenceConfig
    ) -> list[Conversation]:
        """Runs model inference on the provided input.

        Documentation: https://docs.vllm.ai/en/stable/dev/sampling_params.html

        Args:
            input: A list of conversations to run inference on.
            inference_config: Parameters for inference.

        Returns:
            List[Conversation]: Inference output.
        """
        # generation_params = inference_config.generation
        output_conversations = []
        # sampling_params = SamplingParams(
        #    n=1,
        #    max_tokens=generation_params.max_new_tokens,
        #    temperature=generation_params.temperature,
        #    top_p=generation_params.top_p,
        #    frequency_penalty=generation_params.frequency_penalty,
        #    presence_penalty=generation_params.presence_penalty,
        #    stop=generation_params.stop_strings,
        #    stop_token_ids=generation_params.stop_token_ids,
        #    min_p=generation_params.min_p,
        # )

        vllm_conversations = []
        non_skipped_conversations = []
        for conversation in input:
            if not conversation.messages:
                logger.warning("Conversation must have at least one message.")
                continue
            vllm_input = None  # self._convert_conversation_to_vllm_input(conversation)
            vllm_conversations.append(vllm_input)
            non_skipped_conversations.append(conversation)

        if len(vllm_conversations) == 0:
            return []

        # enable_tqdm = len(vllm_conversations) >= 2

        # Note: vLLM performs continuous batching under the hood.
        # We pass all the conversations and let vLLM handle the rest.
        chat_responses = []
        # chat_responses = self._llm.chat(
        #    vllm_conversations,
        #    # sampling_params=sampling_params,
        #    lora_request=self._lora_request,
        #    use_tqdm=enable_tqdm,
        # )

        if False:
            for conversation, chat_response in zip(
                non_skipped_conversations, chat_responses
            ):
                new_messages = [
                    Message(content=message.text, role=Role.ASSISTANT)
                    for message in chat_response.outputs
                    if len(chat_response.outputs) > 0
                ]
                messages = [
                    *conversation.messages,
                    *new_messages,
                ]
                new_conversation = Conversation(
                    messages=messages,
                    metadata=conversation.metadata,
                    conversation_id=conversation.conversation_id,
                )
                output_conversations.append(new_conversation)

        if inference_config.output_path:
            self._save_conversations(
                output_conversations,
                inference_config.output_path,
            )
        return output_conversations

    def infer_online(
        self, input: list[Conversation], inference_config: InferenceConfig
    ) -> list[Conversation]:
        """Runs model inference online.

        Args:
            input: A list of conversations to run inference on.
            inference_config: Parameters for inference.

        Returns:
            List[Conversation]: Inference output.
        """
        return self._infer(input, inference_config)

    def infer_from_file(
        self, input_filepath: str, inference_config: InferenceConfig
    ) -> list[Conversation]:
        """Runs model inference on inputs in the provided file.

        This is a convenience method to prevent boilerplate from asserting the
        existence of input_filepath in the generation_params.

        Args:
            input_filepath: Path to the input file containing prompts for
                generation.
            inference_config: Parameters for inference.

        Returns:
            List[Conversation]: Inference output.
        """
        input = self._read_conversations(input_filepath)
        return self._infer(input, inference_config)

    def get_supported_params(self) -> set[str]:
        """Returns a set of supported generation parameters for this engine."""
        # The params of `SglFunction.run()`:
        return {
            "max_new_tokens",
            "stop",
            "stop_token_ids",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "frequency_penalty",
            "presence_penalty",
            "ignore_eos",
            # The following SGLang params are more esoteric
            # and not enabled yet:
            # - return_logprob
            # - logprob_start_len
            # - top_logprobs_num
            # - return_text_in_logprobs
            # SGLang also has the following system params,
            # which shouldn't be configurable by `oumi` users:
            # - stream
            # - backend
            # - use_thread
        }
