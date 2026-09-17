import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_writer.model import M512, PlaneWriter, R_CHANNELS  # noqa: E402
from shared_writer.training import evaluate, med, rnd, run_clip, to_dev  # noqa: E402
from replacement_writer.packs_rollout import load_pack_rollout  # noqa: E402
from replacement_writer.pack_stream import PackStream  # noqa: E402


def evaluate_rungs(writer, val_packs, device, n_rungs, loader=load_pack_rollout):
    out = {}
    for r in range(n_rungs):
        clips = [loader(p, rung=r) for p in val_packs]
        out[r] = evaluate(writer, clips, device)
    return out


def rung_summary(by_rung):
    return {str(r): {
        "retr_mse_writer_T_med": rnd(
            med([x["retr_mse_writer_T"] for x in rows]), 5),
        "retr_corr_writer_T_med": rnd(
            med([x["retr_corr_writer_T"] for x in rows]), 5),
        "n_queries_med": med([x["n_queries"] for x in rows]),
    } for r, rows in sorted(by_rung.items())}


def headline(by_rung):
    # select checkpoints using the mean of the per-rung median retrieval errors
    per = [med([x["retr_mse_writer_T"] for x in rows])
           for rows in by_rung.values()]
    per = [v for v in per if v is not None]
    return sum(per) / len(per) if per else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", required=True,
                    help="train pack dir(s), comma-separated to train jointly "
                         "(e.g. RE10K + SVID).")
    ap.add_argument("--val-packs", required=True,
                    help="separate validation pack dir(s), comma-separated; "
                         "all --packs clips are used for training.")
    ap.add_argument("--out", default="runs/replacement_writer")
    ap.add_argument("--point-dim", type=int, default=M512["point_dim"])
    ap.add_argument("--point-layers", type=int, default=M512["point_layers"])
    ap.add_argument("--contrib-dim", type=int, default=M512["contrib_dim"])
    ap.add_argument("--reader-hidden", type=int, default=M512["reader_hidden"])
    ap.add_argument("--max-res", type=int, default=M512["max_res"])
    ap.add_argument("--depth", type=int, default=M512["depth"])
    ap.add_argument("--base", type=int, default=M512["base"])
    ap.add_argument("--p-input", action="store_true")
    # save plane-pair ranks in the checkpoint so loading preserves the architecture
    ap.add_argument("--ranks", type=int, nargs=3, default=list(R_CHANNELS))
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--max-steps", type=int, default=500000)
    ap.add_argument("--eval-every-steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--subsample", type=int, default=1000000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stream-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    loader = load_pack_rollout

    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"replacement_seed{a.seed}"

    # keep pack roots separate because clip IDs can overlap between corpora
    def _roots(spec, what):
        rs = [Path(p) for p in str(spec).split(",") if p]
        missing = [r for r in rs if not r.is_dir()]
        if missing:
            raise SystemExit(f"no such {what} pack dir(s): {missing}")
        return rs

    train_paths, val_paths, per_root = [], [], []
    for r in _roots(a.packs, "train"):
        ps = sorted(r.glob("*.pt"))
        if a.limit:
            ps = ps[:a.limit]
        train_paths += ps
        per_root.append(f"{r.name}: {len(ps)} train")
    for r in _roots(a.val_packs, "val"):
        ps = sorted(r.glob("*.pt"))
        val_paths += ps
        per_root.append(f"{r.name}: {len(ps)} val")
    overlap = {p.resolve() for p in train_paths} & {p.resolve() for p in val_paths}
    if overlap:
        raise SystemExit(
            f"{len(overlap)} pack FILE(s) used as BOTH train and val, e.g. "
            f"{[str(p) for p in sorted(overlap)[:3]]}")
    if not train_paths:
        raise SystemExit(f"no train packs found under {a.packs}")
    if not val_paths:
        raise SystemExit(f"no val packs found under {a.val_packs}")

    print("packs  " + "  |  ".join(per_root), flush=True)
    if a.stream_workers == 0:
        print(f"loading {len(train_paths) + len(val_paths)} raw packs ...", flush=True)
    load = lambda p: torch.load(p, map_location="cpu", weights_only=False)  # noqa: E731
    if a.stream_workers > 0:
        print(f"lazy mode: {a.stream_workers} prefetch threads, no eager load",
              flush=True)
        val_raw = list(val_paths)
        train_raw = list(train_paths)
        n_rungs = len(load(train_paths[0])["write_sets"])
    else:
        val_raw = [load(p) for p in val_paths]
        train_raw = [load(p) for p in train_paths]
        n_rungs = len(train_raw[0]["write_sets"])
    if a.stream_workers == 0 and any(
            len(p["write_sets"]) != n_rungs for p in train_raw + val_raw):
        raise ValueError("packs disagree on the number of rungs")
    print(f"train {len(train_raw)}  val {len(val_raw)}  rungs {n_rungs}  "
          f"device {device}  tag {tag}", flush=True)

    writer = PlaneWriter(base=a.base, refine_depth=a.depth,
                         point_dim=a.point_dim, contrib_dim=a.contrib_dim,
                         hidden=a.reader_hidden, p_in_encoder=a.p_input,
                         point_layers=a.point_layers,
                         max_res=a.max_res,
                         ranks=tuple(a.ranks)).to(device)
    n_params = sum(p.numel() for p in writer.parameters())
    opt = torch.optim.Adam(writer.parameters(), lr=a.lr)
    total_steps = a.max_steps or a.epochs * len(train_raw)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    print(f"{n_params/1e6:.2f}M params, {total_steps} steps", flush=True)

    history = []
    best = {"score": float("inf"), "step": None}
    rung_gen = torch.Generator().manual_seed(a.seed + 991)

    def pick_rung():
        return int(torch.randint(0, n_rungs, (1,), generator=rung_gen).item())

    def do_eval(epoch, step):
        by_rung = evaluate_rungs(writer, val_raw, device, n_rungs, loader=loader)
        score = headline(by_rung)
        state = {"model": writer.state_dict(), "args": vars(a),
                 "epoch": epoch, "step": step, "val_score": score,
                 "val_by_rung": rung_summary(by_rung)}
        torch.save(state, out_dir / f"{tag}.ckpt")
        improved = score < best["score"]
        if improved:
            best.update(score=score, step=step)
            torch.save(state, out_dir / f"{tag}_best.ckpt")
        return by_rung, score, improved

    t_start = time.time()
    step, epoch = 0, 0
    while step < total_steps:
        epoch += 1
        ep_loss, ep_n = 0.0, 0
        if a.stream_workers > 0:
            src = ((c for c, _p, _r in PackStream(
                train_raw, loader=loader, n_rungs=n_rungs,
                num_workers=a.stream_workers, shuffle=True,
                seed=a.seed + 991 + epoch)))
        else:
            order = torch.randperm(len(train_raw)).tolist()
            src = (loader(train_raw[i], rung=pick_rung()) for i in order)
        for _clip in src:
            c = to_dev(_clip, device)
            loss = run_clip(writer, c, subsample=a.subsample)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            ep_loss += float(loss)
            ep_n += 1
            step += 1
            if step >= total_steps:
                break
            if a.eval_every_steps and step % a.eval_every_steps == 0:
                by_rung, score, improved = do_eval(epoch, step)
                row = {"epoch": epoch, "step": step,
                       "train_mse_norm": round(ep_loss / max(ep_n, 1), 5),
                       "val_score": rnd(score, 5), "best": improved,
                       "by_rung": rung_summary(by_rung),
                       "wall_s": round(time.time() - t_start, 1)}
                history.append(row)
                print(json.dumps(row), flush=True)
                json.dump({"params": n_params, "args": vars(a),
                           "history": history},
                          open(out_dir / f"{tag}_history.json", "w"), indent=1)

    by_rung, score, improved = do_eval(epoch, step)
    row = {"epoch": epoch, "step": step, "final": True,
           "val_score": rnd(score, 5), "best": improved,
           "by_rung": rung_summary(by_rung),
           "wall_s": round(time.time() - t_start, 1)}
    history.append(row)
    print(json.dumps(row, indent=1), flush=True)
    json.dump({"params": n_params, "args": vars(a), "best": best,
               "history": history},
              open(out_dir / f"{tag}_history.json", "w"), indent=1)
    print(f"best {best}", flush=True)


if __name__ == "__main__":
    main()
