# kevin.py
"""
KevinRewardManager – a minimal reward‑manager for veRL
------------------------------------------------------
• Never looks at `reward_model.ground_truth`
• Reads the JSON string produced by `KernelBenchTool`
• Delegates the scalar calculation to your `compute_score` function
• Stores the scalar on the last valid token of each response
• Optionally prints a few example (prompt, response, score) pairs
"""

from __future__ import annotations
from collections import defaultdict
import sys
import torch
from verl import DataProto
from verl.utils.reward_score import default_compute_score


class KevinRewardManager:
    """Lightweight reward manager that consumes KernelBenchTool JSON."""

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        tool_name: str = "kernel_bench",
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.tool_name = tool_name

    # --------------------------------------------------------------
    # main entry: called by veRL during training / validation
    # --------------------------------------------------------------
    def __call__(self, data: DataProto, *, return_dict: bool = False):
        reward_tensor = torch.zeros_like(
            data.batch["responses"], dtype=torch.float32
        )
        reward_extra_info: dict[str, list] = defaultdict(list)
      
        for i, item in enumerate(data):
            # SGLang rollout stores each tool result under extra_info[tool_name]
            tool_json = (
                item.non_tensor_batch.get("extra_info", {})
                .get(self.tool_name)
            )

            # ---------- scalar reward ----------
            score_out = self.compute_score(
                tool_result=tool_json, data_source=data_source
            )
            if isinstance(score_out, dict):
                reward = score_out["score"]
                for k, v in score_out.items():
                    reward_extra_info[k].append(v)
            else:
                reward = score_out

            # place it on the final token so PPO sees one scalar per sequence
            reward_tensor[i, valid_resp_len - 1] = reward

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor
