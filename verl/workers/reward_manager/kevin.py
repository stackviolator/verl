# kevin_reward_manager.py  – keep on PYTHONPATH
import torch
from collections import defaultdict
from verl import DataProto

class KevinRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score,
                 reward_fn_key="task_id", tool_name="kernel_bench"):
        self.tk, self.show, self.compute = tokenizer, num_examine, compute_score
        self.key, self.tool = reward_fn_key, tool_name

    def __call__(self, data: DataProto, *, return_dict=False):
        R = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        extra = defaultdict(list)

        for i, item in enumerate(data):
            # -------- prompt / response lengths ----------
            am = item.batch["attention_mask"]
            plen = item.batch["prompts"].shape[-1]
            p_ok, r_ok = int(am[:plen].sum()), int(am[plen:].sum())

            # -------- text (for optional logging) --------
            prompt = self.tk.decode(item.batch["prompts"][-p_ok:], skip_special_tokens=True)
            resp   = self.tk.decode(item.batch["responses"][:r_ok],  skip_special_tokens=True)

            print(f"prompt: {prompt}")
            print(f"resp: {resp}")

            # -------- scalar reward ----------------------
            score = self.compute(prompt, resp)
            if isinstance(score, dict):
                for k, v in score.items(): extra[k].append(v)
                score = score["score"]

            R[i, r_ok - 1] = score

        return {"reward_tensor": R, "reward_extra_info": extra} if return_dict else R