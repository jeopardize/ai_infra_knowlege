"""
静态批量处理(Static Batching)Demo
纯 Python 模拟,无 GPU/大模型依赖。

展示静态 batching 的两个核心缺陷:
1. 短板效应:batch 内早完成的请求占位空转,等最慢那个完成才整批释放
2. 新请求无法插队:中途到达的请求 E,即使 batch 里有空位(别人空转),
   也必须等整批 [A,B,C,D] 全部完成释放后,才能开始自己的 prefill

运行: python3 static_batching.py
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
    """模拟 prefill:batch 内并行计算,耗时由最长的 prompt 决定"""
    max_prompt = max(r.prompt_len for r in batch)
    cost = 0.5 + max_prompt * 0.01   # 简化:每 100 token 1s,最少 0.5s
    time.sleep(cost)
    return cost


def static_batching(first_batch, late_request, late_arrival_step):
    """
    first_batch:        第一批请求 [A,B,C,D]
    late_request:       中途到达的请求 E
    late_arrival_step:  E 在第几个 decode step 到达
    """
    print("\n========== 静态批量处理 (Static Batching) ==========\n")

    # ---------- 第一个 batch: [A,B,C,D] ----------
    print(f"[t=0.00s] Batch1 组装完成: {[r.id for r in first_batch]}")

    # prefill
    print(f"[t=0.00s] ---- Batch1 prefill 开始 ----")
    t0 = time.time()
    prefill_cost = fake_prefill(first_batch)
    t = time.time() - t0
    print(f"[t={t:.2f}s] ---- Batch1 prefill 完成 (耗时 {prefill_cost:.2f}s) ----")

    # decode
    step_cost = 0.2
    step = 0
    print(f"[t={t:.2f}s] ---- Batch1 decode 开始 ----")

    e_arrival_t = None    # E 到达时刻
    e = late_request

    while not all(r.done for r in first_batch):
        step += 1
        time.sleep(step_cost)
        t += step_cost

        # E 中途到达,但无法插队,只能排队等
        if step == late_arrival_step:
            e_arrival_t = t
            print(f"[t={t:.2f}s] >>> 新请求 {e.id} 到达! "
                  f"但静态 batching 无法插队,只能排队等 Batch1 整批释放")

        parts = []
        for r in first_batch:
            if r.done:
                parts.append(f"{r.id}空转")
            else:
                r.generated += 1
                if r.generated >= r.target_output:
                    r.done = True
                    parts.append(f"{r.id}完成(占位)")
                else:
                    parts.append(f"{r.id}+1token")
        print(f"[t={t:.2f}s] step {step}:  " + "  ".join(parts))

    print(f"[t={t:.2f}s] ---- Batch1 decode 完成,整批释放 ----")
    batch1_release_t = t

    # ---------- Batch1 释放后,E 才能开始 ----------
    if e_arrival_t is not None:
        e_wait = t - e_arrival_t
        print(f"\n[t={t:.2f}s] >>> {e.id} 终于能开始! "
              f"已白白等了 {e_wait:.2f}s (期间 batch 里其实有空位)")
        print(f"[t={t:.2f}s] Batch2 组装完成: [{e.id}]")

        # E 的 prefill
        print(f"[t={t:.2f}s] ---- Batch2 prefill 开始 ----")
        t0 = time.time()
        e_prefill = fake_prefill([e])
        t += time.time() - t0
        print(f"[t={t:.2f}s] ---- Batch2 prefill 完成 (耗时 {e_prefill:.2f}s) ----")

        # E 的 decode
        print(f"[t={t:.2f}s] ---- Batch2 decode 开始 ----")
        step = 0
        while not e.done:
            step += 1
            time.sleep(step_cost)
            t += step_cost
            e.generated += 1
            if e.generated >= e.target_output:
                e.done = True
                print(f"[t={t:.2f}s] step {step}:  {e.id}完成")
            else:
                print(f"[t={t:.2f}s] step {step}:  {e.id}+1token")
        print(f"[t={t:.2f}s] ---- Batch2 decode 完成 ----")

    # ---------- 统计 ----------
    print(f"\n总耗时: {t:.2f}s")

    print("\n缺陷1 - 短板效应 (batch 内空转,完成后占位):")
    for r in first_batch:
        finish_t = prefill_cost + r.target_output * step_cost
        waste = batch1_release_t - finish_t   # 完成后空转到整批释放
        print(f"  {r.id}: 完成时刻 {finish_t:.2f}s, "
              f"空转到整批释放 {waste:.2f}s")

    if e_arrival_t is not None:
        print(f"\n缺陷2 - 新请求无法插队 (E 的饥饿):")
        print(f"  {e.id}: 到达时刻 {e_arrival_t:.2f}s, "
              f"开始时刻 {batch1_release_t:.2f}s, "
              f"白等 {e_wait:.2f}s")
        print(f"  即使 Batch1 里有请求空转腾出的位置,E 也进不来, "
              f"必须等整批释放")

    print(f"\n对比:静态 batching 的两大毛病")
    print(f"  1. 短板效应 -> A/B/C 空转占位")
    print(f"  2. 新请求饥饿 -> E 即使有空位也插不进,白等 {e_wait:.2f}s")
    print(f"  (continuous batching 正是解决这两点:step 级动态进出)")


if __name__ == "__main__":
    # 第一批:4 个请求,长度拉开,凸显短板效应
    first_batch = [
        Request(id="A", prompt_len=20, target_output=2),   # 很快完成
        Request(id="B", prompt_len=30, target_output=3),
        Request(id="C", prompt_len=40, target_output=5),
        Request(id="D", prompt_len=50, target_output=8),   # 最慢,决定整批耗时
    ]
    # 中途到达的请求 E(在 step 2 到达,自己其实很快,但被迫等)
    late_request = Request(id="E", prompt_len=25, target_output=2)
    static_batching(first_batch, late_request, late_arrival_step=2)
