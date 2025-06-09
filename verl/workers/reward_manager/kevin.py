# kevin_reward_manager.py
"""
KevinRewardManager – a minimal reward‑manager for veRL
------------------------------------------------------
• Never looks at `reward_model.ground_truth`
• Reads the JSON string produced by `KernelBenchTool`
• Delegates the scalar calculation to your `compute_score` function
• Stores the scalar on the last valid token of each response
• Optionally prints a few example (prompt, response, score) pairs
To use:
    custom_reward_function.path=kevin_reward.py         # <-- your compute_score
    custom_reward_function.name=compute_score
    reward_model.enable=False
    reward_model.reward_manager=kevin                   # any name ≠ built‑ins
and make sure this file is importable (PYTHONPATH or a pip‑installable package).
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
        tool_name: str = "kernel_bench",  # matches KernelBenchTool.name
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

        printed = {}  # how many examples we have logged per data_source

        print(data)

        for i, item in enumerate(data):
            print(f"=== Processing item {i} ===")
            print(f"item.non_tensor_batch keys: {list(item.non_tensor_batch.keys())}")
            print(f"item.non_tensor_batch: {item.non_tensor_batch}")

            print(f"meta: {item.non_tensor_batch['meta']}")
            
            # ---------- slice out prompt / response ----------
            amask = item.batch["attention_mask"]
            plen = item.batch["prompts"].shape[-1]
            valid_prompt_len = int(amask[:plen].sum())
            valid_resp_len = int(amask[plen:].sum())

            prompt_ids = item.batch["prompts"][-valid_prompt_len:]
            resp_ids = item.batch["responses"][:valid_resp_len]

            prompt_str = self.tokenizer.decode(
                prompt_ids, skip_special_tokens=True
            )
            resp_str = self.tokenizer.decode(
                resp_ids, skip_special_tokens=True
            )

            # ---------- metadata ----------
            # Handle missing data_source key gracefully
            if self.reward_fn_key in item.non_tensor_batch:
                data_source = item.non_tensor_batch[self.reward_fn_key]
            else:
                print(f"Warning: '{self.reward_fn_key}' not found in non_tensor_batch. Available keys: {list(item.non_tensor_batch.keys())}")
                # Try some common fallback strategies
                possible_keys = ["data_source", "dataset", "source", "task", "dataset_name"]
                data_source = None
                for key in possible_keys:
                    if key in item.non_tensor_batch:
                        data_source = item.non_tensor_batch[key]
                        print(f"Using fallback key '{key}' as data_source: {data_source}")
                        break
                
                if data_source is None:
                    # Try to derive from available metadata
                    if "task_id" in item.non_tensor_batch:
                        task_id = item.non_tensor_batch["task_id"]
                        data_source = f"kernel_bench_{task_id}"
                        print(f"Derived data_source from task_id: {data_source}")
                    elif "meta" in item.non_tensor_batch:
                        meta = item.non_tensor_batch["meta"]
                        if isinstance(meta, dict) and "name" in meta:
                            data_source = f"kernel_bench_{meta['name']}"
                            print(f"Derived data_source from meta.name: {data_source}")
                        else:
                            data_source = "kernel_bench"
                            print(f"Using generic data_source: {data_source}")
                    else:
                        data_source = "kernel_bench"  # Default for kernel benchmarking
                        print(f"Using default data_source: {data_source}")

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

            # ---------- debug prints ----------
            if printed.get(data_source, 0) < self.num_examine:
                printed[data_source] = printed.get(data_source, 0) + 1
                print("=" * 40, f"[{data_source} example]")
                print("[prompt]\n", prompt_str)
                print("[response]\n", resp_str)
                print("[tool_json]\n", tool_json)
                print("[score]", reward)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor