"""
动态批量处理(Dynamic Batching)Demo
纯 Python 模拟,无 GPU/大模型依赖。

展示动态 batching 的三个核心特征:
1. req 有人完成之后，可以将新的填入 batch

运行: python3 dynamic_batching.py
"""

import time
from dataclasses import dataclass


@dataclass
class Request:
    id: str
    prompt_len: int        # prompt 长度(token),影响 prefill 耗时
    target_output: int     # 期望输出 token 数,影响 decode 步数
    generated: int = 0     # 已生成 token 数
    done: bool = False     # 是否完成


def fake_prefill(batch):
    
    max_prompt = max(r.prompt_len for r in batch)
    cost = 0.5 + max_prompt * 0.01   # 简化:每 100 token 1s,最少 0.5s
    time.sleep(cost)
    return cost


def dynamic_batching(requests):
    print("\n========== 动态批量处理 (Dynamic Batching) ==========\n")

    # 1. 凑批
    print(f"[t=0.00s] Batch 组装完成: {[r.id for r in requests]}")

    # 2. prefill(整批一起算)
    print(f"[t=0.00s] ---- prefill 开始 ----")
    t0 = time.time()
    prefill_cost = fake_prefill(requests)
    t = time.time() - t0
    print(f"[t={t:.2f}s] ---- prefill 完成 (耗时 {prefill_cost:.2f}s) ----")

    # 3. decode(每个 step 整批同步推进)
    step_cost = 0.2   # 每个 decode step 固定 0.2s
    step = 0
    print(f"[t={t:.2f}s] ---- decode 开始 ----")

    while not all(r.done for r in requests):
        step += 1
        time.sleep(step_cost)
        t += step_cost

        parts = []
        for r in requests:
            if r.done:
                # 已完成,但仍占着 batch 位置,什么也不做
                parts.append(f"{r.id}空转")
            else:
                r.generated += 1
                if r.generated >= r.target_output:
                    r.done = True
                    parts.append(f"{r.id}完成(占位)")
                else:
                    parts.append(f"{r.id}+1token")
        print(f"[t={t:.2f}s] step {step}:  " + "  ".join(parts))

    print(f"[t={t:.2f}s] ---- decode 完成 ----")

    # 4. 统计:短板效应
    print(f"\n总耗时: {t:.2f}s  (由最长的请求 D 决定,即短板)")
    print("\n各请求的空转/等待情况:")
    for r in requests:
        # 该请求"理想单独耗时"= 自己的 prefill + 自己的 decode
        ideal_prefill = 0.5 + r.prompt_len * 0.01
        ideal_decode = r.target_output * step_cost
        ideal = ideal_prefill + ideal_decode
        waste = t - ideal
        print(f"  {r.id}: 理想单独耗时 {ideal:.2f}s, "
              f"在 batch 中空转/等待 {waste:.2f}s")

    # 5. 对比:如果串行跑各请求
    serial_total = sum(0.5 + r.prompt_len * 0.01 + r.target_output * step_cost
                       for r in requests)
    print(f"\n对比:若串行处理 4 个请求,总耗时 {serial_total:.2f}s")
    print(f"     动态 batching 总耗时 {t:.2f}s (并行收益)")
    print(f"     但短请求被迫等长请求,产生空转浪费 (短板效应)")


if __name__ == "__main__":
    # 4 个请求,长度故意拉开差距,凸显短板效应
    requests = [
        Request(id="A", prompt_len=20, target_output=2),   # 很快完成
        Request(id="B", prompt_len=30, target_output=3),
        Request(id="C", prompt_len=40, target_output=5),
        Request(id="D", prompt_len=50, target_output=8),   # 最慢,决定整批耗时
    ]
    dynamic_batching(requests)
