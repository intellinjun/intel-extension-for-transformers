#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The neural engine operator mapping file."""

from .op import Operator, operator_registry
from .tensor import Tensor

# Fused_op size + add + slice
# This pattern is used for dealing with position_mask originally in bert model
@operator_registry(operator_type='SlicePositionIds')
class SlicePositionIds(Operator):
    """Register the SlicePositionIds operator."""

    def __init__(self):
        """The init function of this operator."""
        super().__init__()

    def set_attr(self, framework, node):
        """Extract the node attr."""
        if framework == "torch":
            self._op_type = "Slice"
            self._attr['axes'] = node.inputsAt(1).toIValue()
            self._attr['starts'] = node.inputsAt(2).toIValue()
            self._attr['ends_with_tensor'] = 1
            self._attr['steps'] = node.inputsAt(4).toIValue()

