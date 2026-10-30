# -*- coding: utf-8 -*-
"""
以小验大 Step2: 用数据集访问开启 --enable-return-routed-experts 的 vLLM 服务，
把响应中附带的 routed experts 落盘成 stat_moe.py 可统计的逐层 txt。

输出目录内每个 MoE 层一个 `layers.<层号>.txt`，每行一个 token 的 top-K 专家 ID
（逗号分隔）。统计与裁剪衔接:

    python scripts/collect_routed_experts.py \
        --api-base http://127.0.0.1:8006 \
        --model DeepSeek-V4-Flash-bf16 \
        --dataset data/countbenchqa.csv --column Question \
        --model-config /path/to/model/config.json \
        --output-dir run_dir/topk_ids_by_layer

    python scripts/stat_moe.py --top-k 64 --log-dir run_dir/topk_ids_by_layer
    # top64_experts.json 生成在 run_dir/ 下（--log-dir 的上级目录）

响应中 routed_experts 的形态按实现可能是:
  * base64 字符串 -> 按 --dtype 解码后 reshape 为 (T, L, K)
  * 扁平 int 列表 / 嵌套列表 -> 同样 reshape 为 (T, L, K)
其中 L=MoE 层数、K=每 token 激活专家数（默认从 --model-config 的 config.json
推断，可用 --num-moe-layers/--experts-per-token 显式覆盖）。首个请求会打印
形状核对信息（T 与 prompt+completion token 数的关系），请务必抽样核对后再
放量采集；层号与模型全局层号的对应关系可用 --layer-offset/--layer-indices 校正。
"""

import argparse
import array
import base64
import csv
import json
import os
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

DTYPES = {
    "int8": "b", "uint8": "B", "int16": "h",
    "int32": "i", "int64": "q",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="采集 vLLM 响应中的 routed experts（以小验大 Step2）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 服务
    p.add_argument("--api-base", required=True,
                   help="服务地址，如 http://127.0.0.1:8006")
    p.add_argument("--model", default=None,
                   help="served-model-name；缺省时查询 /v1/models 取第一个")
    p.add_argument("--api-key", default=None, help="可选 Bearer token")
    p.add_argument("--endpoint", choices=["chat", "completions"], default="chat",
                   help="走 /v1/chat/completions 或 /v1/completions")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--no-verify-ssl", action="store_true")

    # 生成参数
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--extra-body", default=None,
                   help='附加 JSON 请求体字段, 如 \'{"top_k":1}\'')

    # 数据集
    p.add_argument("--dataset", required=True,
                   help="数据集文件: .csv / .jsonl / .json(list) / .txt(每行一条)")
    p.add_argument("--column", default="Question",
                   help="CSV 问题列名（或 0-based 列序号）")
    p.add_argument("--field", default="question",
                   help="jsonl/json 中问题字段（支持 a.b 点路径）")
    p.add_argument("--limit", type=int, default=None, help="只取前 N 条")

    # routed experts 解析
    p.add_argument("--model-config", default=None,
                   help="模型 config.json 路径，用于自动推断 L(层数)/K(每token专家数)")
    p.add_argument("--num-moe-layers", type=int, default=None,
                   help="响应数组中的层数 L；不给则从 --model-config 推断")
    p.add_argument("--experts-per-token", type=int, default=None,
                   help="每 token 激活专家数 K；不给则从 --model-config 推断")
    p.add_argument("--dtype", choices=list(DTYPES), default="int32",
                   help="base64 解码用 dtype")
    p.add_argument("--layer-offset", type=int, default=0,
                   help="数组第 i 层 -> 全局层号 offset+i")
    p.add_argument("--layer-indices", default=None,
                   help="显式给出数组各层对应的全局层号, 逗号分隔, 如 '0,1,2,4,5'")

    # 输出
    p.add_argument("--output-dir", required=True,
                   help="逐层 txt 输出目录（建议命名为 run_dir/topk_ids_by_layer）")
    p.add_argument("--raw-dir", default=None,
                   help="可选: 保存原始响应 JSONL，便于排查响应格式")
    return p.parse_args()


def load_prompts(args):
    path = args.dataset
    ext = os.path.splitext(path)[1].lower()
    prompts = []
    if ext == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            raise SystemExit(f"CSV 为空: {path}")
        header = rows[0]
        col = args.column
        if col.isdigit():
            idx = int(col)
        elif col in header:
            idx = header.index(col)
        else:
            raise SystemExit(
                f"CSV 中找不到列 {col!r}; 表头: {header}。"
                f"注意: 列名区分大小写, 也可用 0-based 序号")
        prompts = [r[idx] for r in rows[1:] if len(r) > idx and r[idx].strip()]
    elif ext in (".jsonl", ".json"):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            if ext == ".jsonl":
                items = [json.loads(line) for line in f if line.strip()]
            else:
                items = json.load(f)
        for it in items:
            cur = it
            for key in args.field.split("."):
                cur = cur[key]
            if isinstance(cur, str) and cur.strip():
                prompts.append(cur)
    else:
        with open(path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    if args.limit:
        prompts = prompts[:args.limit]
    if not prompts:
        raise SystemExit(f"未能从 {path} 解析出任何 prompt")
    return prompts


def infer_lk(args):
    """从模型 config.json 推断 (L, K)；推断不了则要求显式指定。"""
    L, K, source = args.num_moe_layers, args.experts_per_token, "cli"
    if (L is None or K is None) and args.model_config:
        with open(args.model_config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # 兼容嵌套 text_config
        text_cfg = cfg.get("text_config", cfg)
        if L is None:
            n_layers = text_cfg.get("num_hidden_layers")
            if n_layers is not None:
                # DeepSeek 系: 前 first_k_dense_replace 层为 dense, 无 routed experts
                dense = text_cfg.get("first_k_dense_replace", 0)
                L = n_layers - dense
                source = f"model-config(num_hidden_layers={n_layers}, first_k_dense_replace={dense})"
        if K is None:
            for key in ("num_experts_per_tok", "n_activated_experts",
                        "experts_per_token", "n_experts_per_token"):
                if key in text_cfg:
                    K = int(text_cfg[key])
                    source = f"model-config({key}={K})"
                    break
    if L is None or K is None:
        raise SystemExit(
            "无法确定响应数组形状 (T, L, K): 请通过 --num-moe-layers/--experts-per-token "
            "显式指定, 或提供 --model-config。"
            f"当前 L={L}, K={K}")
    return L, K, source


def find_routed_experts(obj):
    """在响应 JSON 中递归查找 routed_experts 字段。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "routed_experts" and v is not None:
                return v
            found = find_routed_experts(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_routed_experts(v)
            if found is not None:
                return found
    return None


_write_lock = threading.Lock()


def run_one(session_state, prompt, args, L, K):
    body = {
        "model": session_state["model"],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": False,
    }
    if args.endpoint == "chat":
        body["messages"] = [{"role": "user", "content": prompt}]
    else:
        body["prompt"] = prompt
    if args.extra_body:
        body.update(json.loads(args.extra_body))

    url = args.api_base.rstrip("/") + (
        "/v1/chat/completions" if args.endpoint == "chat" else "/v1/completions")
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    last_err = None
    for attempt in range(1, args.retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            resp = urllib.request.urlopen(req, timeout=args.timeout,
                                          context=session_state["ssl_ctx"])
            payload = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:300]
            if e.code < 500 and e.code != 429:
                raise RuntimeError(f"HTTP {e.code}: {detail}")
            last_err = RuntimeError(f"HTTP {e.code}: {detail}")
        except Exception as e:  # 网络类错误重试
            last_err = e
        if attempt < args.retries:
            time.sleep(min(30.0, 2 ** attempt))
    else:
        raise last_err if last_err else RuntimeError("请求失败")

    routed = find_routed_experts(payload)
    usage = payload.get("usage", {}) or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")

    if session_state["raw_dir"]:
        with _write_lock:
            with open(os.path.join(session_state["raw_dir"], "responses.jsonl"),
                      "a", encoding="utf-8") as f:
                f.write(json.dumps({"prompt": prompt[:200], "response": payload},
                                   ensure_ascii=False) + "\n")
    if routed is None:
        raise ValueError(
            "响应中未找到 routed_experts 字段——确认服务以 "
            "--enable-return-routed-experts 启动, 且该版本会把字段放进响应"
            "(可用 --raw-dir 保存响应排查)")

    dtype_code = DTYPES[args.dtype]
    # 统一转 (T, L, K) 展平数组
    if isinstance(routed, str):
        raw = base64.b64decode(routed)
        buf = array.array(dtype_code)
        buf.frombytes(raw[:len(raw) - (len(raw) % buf.itemsize)])
        flat = buf.tolist()
    elif isinstance(routed, list):
        if routed and isinstance(routed[0], list):
            flat = []
            for tok in routed:
                for layer in tok:
                    flat.extend(layer if isinstance(layer, list) else [layer])
        else:
            flat = [int(x) for x in routed]
    else:
        raise ValueError(f"无法解析的 routed_experts 类型: {type(routed)}")

    numel = len(flat)
    if numel == 0 or numel % (L * K) != 0:
        raise ValueError(
            f"routed_experts 元素数 {numel} 不是 L*K={L * K} 的整数倍, "
            f"请核对 --num-moe-layers/--experts-per-token/--dtype; "
            f"建议用 --raw-dir 保存响应核对实际形状")
    T = numel // (L * K)

    # 逐层追加写 txt: 每行一个 token 的 K 个专家 ID
    if args.layer_indices:
        layer_ids = [int(x) for x in args.layer_indices.split(",")]
        if len(layer_ids) != L:
            raise ValueError(
                f"--layer-indices 给了 {len(layer_ids)} 个层号, 与 L={L} 不一致")
    else:
        layer_ids = [args.layer_offset + i for i in range(L)]

    files = session_state["files"]
    with _write_lock:
        for li, global_idx in enumerate(layer_ids):
            path = files.setdefault(global_idx, os.path.join(
                args.output_dir, f"layers.{global_idx}.txt"))
            with open(path, "a", encoding="utf-8") as f:
                for t in range(T):
                    row = flat[t * L * K + li * K: t * L * K + (li + 1) * K]
                    f.write(",".join(str(int(e)) for e in row) + "\n")

    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "T": T, "numel": numel}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    raw_dir = None
    if args.raw_dir:
        raw_dir = args.raw_dir
        os.makedirs(raw_dir, exist_ok=True)

    ssl_ctx = None
    if args.no_verify_ssl:
        ssl_ctx = ssl._create_unverified_context()

    prompts = load_prompts(args)
    L, K, lk_source = infer_lk(args)

    model = args.model
    if model is None:
        with urllib.request.urlopen(args.api_base.rstrip("/") + "/v1/models",
                                    timeout=30, context=ssl_ctx) as r:
            model = json.loads(r.read().decode("utf-8"))["data"][0]["id"]
        print(f"[collect] 未指定 --model, 使用服务端第一个模型: {model}")

    session_state = {"model": model, "ssl_ctx": ssl_ctx, "raw_dir": raw_dir,
                     "files": {}}

    print(f"[collect] 数据集 {args.dataset}: {len(prompts)} 条 prompt")
    print(f"[collect] 数组形状 (T, L={L}, K={K}), 来源: {lk_source}")
    print(f"[collect] 输出目录: {args.output_dir}")

    results, failures = [], []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(run_one, session_state, p, args, L, K): i
                   for i, p in enumerate(prompts)}
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            done += 1
            try:
                info = fut.result()
                results.append((idx, info))
                if done == 1 or done % 50 == 0 or done == len(prompts):
                    print(f"[collect] {done}/{len(prompts)} 完成, "
                          f"最近请求 T={info['T']} "
                          f"(prompt={info['prompt_tokens']}, "
                          f"completion={info['completion_tokens']})")
            except Exception as e:
                failures.append((idx, str(e)))
                print(f"[collect] 第 {idx} 条失败: {e}", file=sys.stderr)

    elapsed = time.time() - t0
    total_T = sum(r[1]["T"] for r in results)
    manifest = {
        "dataset": os.path.abspath(args.dataset),
        "prompts_total": len(prompts),
        "ok": len(results), "failed": len(failures),
        "L": L, "K": K, "lk_source": lk_source, "dtype": args.dtype,
        "layer_offset": args.layer_offset, "layer_indices": args.layer_indices,
        "total_tokens_counted": total_T,
        "layer_files": {str(k): os.path.basename(v)
                        for k, v in sorted(session_state["files"].items())},
        "elapsed_sec": round(elapsed, 1),
        "failures": [{"idx": i, "error": m[:300]} for i, m in failures[:50]],
    }
    with open(os.path.join(args.output_dir, "_collect_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[collect] 完成: 成功 {len(results)} / 失败 {len(failures)}, "
          f"计入统计 token 行数 {total_T}, 耗时 {elapsed:.0f}s")
    print(f"[collect] 逐层文件: {sorted(session_state['files'].values())}")
    print(f"[collect] manifest: {os.path.join(args.output_dir, '_collect_manifest.json')}")
    if failures:
        print(f"[collect] 有失败请求, 详见 manifest.failures; "
              f"失败比例高时不要直接用于统计", file=sys.stderr)
        sys.exit(2)
    if done == 1:
        print("[collect] 首个请求形状核对: T 应约等于 prompt+completion-1 或 "
              "completion-1 (取决于实现丢弃哪个 token 的路由); "
              "若明显不符, 请核对 L/K/dtype 后重采")


if __name__ == "__main__":
    main()
