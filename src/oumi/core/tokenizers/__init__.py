# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tokenizers module for the Oumi (Open Universal Machine Intelligence) library.

This module provides base classes for tokenizers used in the Oumi framework.
These base classes serve as foundations for creating custom tokenizers for various
natural language processing tasks.
"""

from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.tokenizers.special_tokens import get_default_special_tokens

__all__ = [
    "BaseTokenizer",
    "get_default_special_tokens",
]
