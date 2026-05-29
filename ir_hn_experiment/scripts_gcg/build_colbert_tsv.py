import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    collection = load_jsonl(dataset_dir / "collection.jsonl")
    train_rows = load_jsonl(dataset_dir / f"{args.split}.jsonl")
    queries_rows = load_jsonl(dataset_dir / f"queries_{args.split}.jsonl")

    qmap = {str(x["qid"]): x["query"] for x in queries_rows}

    with (out_dir / "queries.tsv").open("w", encoding="utf-8") as fq:
        for qid, query in sorted(qmap.items(), key=lambda x: x[0]):
            fq.write(f"{qid}\t{query}\n")

    with (out_dir / "collection.tsv").open("w", encoding="utf-8") as fc:
        for row in collection:
            pid = str(row["pid"])
            text = ((row.get("title", "") or "") + " " + (row.get("text", "") or "")).strip()
            fc.write(f"{pid}\t{text}\n")

    with (out_dir / "triples.tsv").open("w", encoding="utf-8") as ft:
        for row in train_rows:
            qid = str(row["qid"])
            for pos_id in row["pos_ids"]:
                for neg_id in row["neg_ids"]:
                    ft.write(f"{qid}\t{pos_id}\t{neg_id}\n")

    print(f"saved tsvs to: {out_dir}")


if __name__ == "__main__":
    main()
