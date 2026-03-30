import argparse
import timeit
import statistics
import torch

from cs336_basics.model import BasicsTransformerLM

MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

def build_model(model_size: str, context_length: int, vocab_size: int = 10000):
    cfg = MODEL_CONFIGS[model_size]
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=10000,
    ).cuda()
    return model

def run_step(model, input_ids, mode: str):
    if mode == "forward":
        with torch.no_grad():
            _ = model(input_ids)
    elif mode == "forward_backward":
        logits = model(input_ids)
        loss = logits.mean()
        loss.backward()
    else:
        raise ValueError(f"Unknown mode: {mode}")

def benchmark(model_size, context_length, batch_size, num_warmup, num_iters, mode):
    torch.cuda.empty_cache()

    model = build_model(model_size, context_length)
    model.train() if mode == "forward_backward" else model.eval()

    input_ids = torch.randint(
        0, 10000, (batch_size, context_length), device="cuda"
    )

    # warmup
    for _ in range(num_warmup):
        if mode == "forward_backward":
            model.zero_grad(set_to_none=True)
        run_step(model, input_ids, mode)
        torch.cuda.synchronize()

    times = []
    for _ in range(num_iters):
        if mode == "forward_backward":
            model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        start = timeit.default_timer()

        run_step(model, input_ids, mode)

        torch.cuda.synchronize()
        end = timeit.default_timer()

        times.append(end - start)

    mean_t = statistics.mean(times)
    std_t = statistics.stdev(times) if len(times) > 1 else 0.0

    print(f"model={model_size}")
    print(f"context_length={context_length}")
    print(f"mode={mode}")
    print(f"num_warmup={num_warmup}, num_iters={num_iters}")
    print(f"mean={mean_t:.6f} s")
    print(f"std={std_t:.6f} s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, required=True, choices=MODEL_CONFIGS.keys())
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_warmup", type=int, default=5)
    parser.add_argument("--num_iters", type=int, default=10)
    parser.add_argument("--mode", type=str, default="forward_backward",
                        choices=["forward", "forward_backward"])
    args = parser.parse_args()

    benchmark(
        model_size=args.model_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
        mode=args.mode,
    )

if __name__ == "__main__":
    main()