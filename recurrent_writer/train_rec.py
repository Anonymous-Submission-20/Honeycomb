import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared_writer.model import M512, R_CHANNELS, PlaneWriter  # noqa: E402
from shared_writer.training import rnd  # noqa: E402
from recurrent_writer.fusion import FUSION_HIDDEN  # noqa: E402
from recurrent_writer.incremental import BOUNDS_POLICY  # noqa: E402
from recurrent_writer.incremental import IncrementalWriter  # noqa: E402
from recurrent_writer.recurrence import (  # noqa: E402
    initial_frame, pack_chunks, pack_queries)
from recurrent_writer.time_map import TIME_MAP, TIME_MAPS, make_time_map  # noqa: E402
from replacement_writer.pack_stream import PackStream, pack_paths  # noqa: E402

Q_KEYS = ("points_world", "viewdirs", "origins_world", "times", "mem_norm", "frame")
C_KEYS = ("points_world", "times", "feats", "viewdirs", "origins_world")


def to_dev(d, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in d.items()}


def prepare(pack, time_map, chunk_span):
    chunks = pack_chunks(pack)
    queries = [pack_queries(pack, rung=k) for k in range(len(chunks))]
    return chunks, queries, initial_frame(pack, time_map, chunk_span=chunk_span)


def _subsample(q, n, gen):
    total = q["mem_norm"].shape[0]
    if not n or total <= n:
        return q
    idx = torch.randperm(total, generator=gen, device=q["mem_norm"].device)[:n]
    return {k: (v[idx] if torch.is_tensor(v) and v.shape[:1] == (total,) else v)
            for k, v in q.items()}


def run_clip(model, chunks, queries, frame, device, subsample=0, gen=None):
    # supervise retrieval after every write
    state, loss, n = None, 0.0, 0
    for k, c in enumerate(chunks):
        state = model.update(state, to_dev(c, device),
                             frame=frame if state is None else None)
        q = to_dev(queries[k], device)
        if subsample:
            q = _subsample(q, subsample, gen)
        pred = model.query(state, q["points_world"], q["times"],
                           q["viewdirs"], q["origins_world"])
        loss = loss + ((pred - q["mem_norm"]) ** 2).mean()
        n += 1
    return loss / max(n, 1)


@torch.no_grad()
def evaluate(model, packs, device, time_map, chunk_span):
    n_writes = len(packs[0]["write_sets"])
    acc = {k: {"all": [], "old": [], "new": [],
               "all_raw": [], "old_raw": [], "new_raw": []}
           for k in range(n_writes)}
    for pk in packs:
        chunks, queries, frame = prepare(pk, time_map, chunk_span)
        state = None
        for k, c in enumerate(chunks):
            state = model.update(state, to_dev(c, device),
                                 frame=frame if state is None else None)
            q = to_dev(queries[k], device)
            pred = model.query(state, q["points_world"], q["times"],
                               q["viewdirs"], q["origins_world"])
            diff = pred - q["mem_norm"]
            se = (diff ** 2).mean(dim=1)
            se_raw = ((diff * q["std"]) ** 2).mean(dim=1)
            acc[k]["all"].append(float(se.mean()))
            acc[k]["all_raw"].append(float(se_raw.mean()))
            # old content means points written before this chunk, regardless of query view
            first_new = min(c["latent_times"])
            old = q["times"] < first_new
            if bool(old.any()):
                acc[k]["old"].append(float(se[old].mean()))
                acc[k]["old_raw"].append(float(se_raw[old].mean()))
            new = ~old
            if bool(new.any()):
                acc[k]["new"].append(float(se[new].mean()))
                acc[k]["new_raw"].append(float(se_raw[new].mean()))
    def med(xs):
        if not xs:
            return None
        s = sorted(xs)
        return s[len(s) // 2]
    return {str(k): {"all_med": rnd(med(v["all"]), 5),
                     "old_med": rnd(med(v["old"]), 5),
                     "new_med": rnd(med(v["new"]), 5),
                     "all_raw_med": rnd(med(v["all_raw"]), 6),
                     "old_raw_med": rnd(med(v["old_raw"]), 6),
                     "new_raw_med": rnd(med(v["new_raw"]), 6),
                     "n_clips": len(v["all"])} for k, v in acc.items()}


def headline(summary):
    # select checkpoints using the mean of the per-write median errors in raw latent units
    vals = [v["all_raw_med"] for v in summary.values()
            if v.get("all_raw_med") is not None]
    return sum(vals) / len(vals) if vals else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", required=True)
    ap.add_argument("--val-packs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fusion", choices=("pool", "learned"), default="learned")
    ap.add_argument("--time-map", choices=sorted(TIME_MAPS), default=TIME_MAP)
    ap.add_argument("--chunk-span", type=float, default=8.0)
    ap.add_argument("--fusion-hidden", type=int, default=FUSION_HIDDEN)
    ap.add_argument("--bounds-policy", choices=("tight", "reserve"), default=BOUNDS_POLICY)
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
    ap.add_argument("--lr", type=float, default=5e-4)
    # minimum learning rate at the end of the cosine schedule
    ap.add_argument("--lr-min", type=float, default=0.0,
                    help="cosine eta_min; 0.0 = decay to zero")
    ap.add_argument("--max-steps", type=int, default=500000)
    ap.add_argument("--eval-every-steps", type=int, default=10000)
    ap.add_argument("--subsample", type=int, default=200000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stream-workers", type=int, default=8)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tm = make_time_map(a.time_map, chunk_span=a.chunk_span)

    def load_dir(spec, limit):
        roots = [Path(x) for x in str(spec).split(",") if x]
        missing = [r for r in roots if not r.is_dir()]
        if missing:
            raise SystemExit(f"no such pack dir(s): {missing}")
        out = []
        for r in roots:
            ps = sorted(r.glob("*.pt"))
            if limit:
                ps = ps[:limit]
            if not ps:
                raise SystemExit(f"no packs under {r}")
            print(f"  {r.name}: {len(ps)} packs", flush=True)
            out += [torch.load(q, map_location="cpu", weights_only=False) for q in ps]
        return out

    train_paths = pack_paths([Path(x) for x in str(a.packs).split(",") if x])
    if a.limit:
        train_paths = train_paths[:a.limit]
    val = load_dir(a.val_packs, a.val_limit)
    n_writes = len(val[0]["write_sets"])

    def _prep(raw, rung=0):
        return prepare(raw, tm, a.chunk_span)

    stream = PackStream(train_paths, loader=_prep, rung_fn=lambda i: 0,
                        num_workers=a.stream_workers, shuffle=True,
                        seed=a.seed + 123)
    train = train_paths
    print(f"train {len(train)}  val {len(val)}  writes {n_writes}  "
          f"fusion {a.fusion}  time_map {a.time_map}  device {device}  tag {a.tag}",
          flush=True)

    writer = PlaneWriter(base=a.base, refine_depth=a.depth, point_dim=a.point_dim,
                         contrib_dim=a.contrib_dim, hidden=a.reader_hidden,
                         p_in_encoder=a.p_input, point_layers=a.point_layers,
                         max_res=a.max_res,
                         ranks=tuple(a.ranks))
    model = IncrementalWriter(writer, fusion=a.fusion, hidden=a.fusion_hidden,
                              bounds_policy=a.bounds_policy).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=a.max_steps, eta_min=a.lr_min)
    print(f"{n_params/1e6:.2f}M params, {a.max_steps} steps, "
          f"lr {a.lr} -> {a.lr_min}", flush=True)

    history, best = [], {"score": float("inf"), "step": None}
    gen = torch.Generator(device=device).manual_seed(a.seed + 991)

    def do_eval(epoch, step):
        model.eval()
        summary = evaluate(model, val, device, tm, a.chunk_span)
        model.train()
        score = headline(summary)
        state = {"model": model.state_dict(), "args": vars(a),
                 "epoch": epoch, "step": step, "val_score": score,
                 "val_by_write": summary}
        torch.save(state, out_dir / f"{a.tag}.ckpt")
        improved = score < best["score"]
        if improved:
            best.update(score=score, step=step)
            torch.save(state, out_dir / f"{a.tag}_best.ckpt")
        return summary, score, improved

    t0 = time.time()
    step, epoch = 0, 0
    while step < a.max_steps:
        epoch += 1
        ep_loss, ep_n = 0.0, 0
        for prepared, _path, _rung in stream:
            chunks, queries, frame = prepared
            loss = run_clip(model, chunks, queries, frame, device,
                            subsample=a.subsample, gen=gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            ep_loss += float(loss)
            ep_n += 1
            step += 1
            if step >= a.max_steps:
                break
            if a.eval_every_steps and step % a.eval_every_steps == 0:
                summary, score, improved = do_eval(epoch, step)
                row = {"epoch": epoch, "step": step,
                       "train_mse_norm": rnd(ep_loss / max(ep_n, 1), 5),
                       "val_score": rnd(score, 5), "best": improved,
                       "by_write": summary,
                       "wall_s": round(time.time() - t0, 1)}
                history.append(row)
                print(json.dumps(row), flush=True)
                json.dump({"params": n_params, "args": vars(a),
                           "history": history},
                          open(out_dir / f"{a.tag}_history.json", "w"), indent=1)

    summary, score, improved = do_eval(epoch, step)
    row = {"epoch": epoch, "step": step, "final": True,
           "val_score": rnd(score, 5), "best": improved, "by_write": summary,
           "wall_s": round(time.time() - t0, 1)}
    history.append(row)
    print(json.dumps(row), flush=True)
    json.dump({"params": n_params, "args": vars(a), "history": history},
              open(out_dir / f"{a.tag}_history.json", "w"), indent=1)
    print(f"best {best}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
