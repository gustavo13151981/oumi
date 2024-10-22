from typing import Any, Dict

from click import prompt

from oumi.core.datasets.base_dpo_dataset import BaseExperimentalDpoDataset
from oumi.core.types.conversation import Conversation, Message, Role


class OrcaDpoPairsDataset(BaseExperimentalDpoDataset):
    """Dataset class for the Intel/orca_dpo_pairs dataset."""

    def transform_preference(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform the sample into the required format."""
        system_prompt = sample.get("system", "")
        question = sample["question"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        chosen_conversation = self._create_conversation(system_prompt, question, chosen)
        rejected_conversation = self._create_conversation(
            system_prompt, question, rejected
        )

        return {
            "prompt": prompt,
            "chosen": chosen_conversation,
            "rejected": rejected_conversation,
        }

    def _create_conversation(
        self, system_prompt: str, request: str, response: str
    ) -> Conversation:
        """Create a Conversation object from prompt and response."""
        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=request),
            Message(role=Role.ASSISTANT, content=response),
        ]
        return Conversation(messages=messages)
