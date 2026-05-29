import argparse
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer, Indexer, Searcher
from colbert.data import Queries


def train(args):
    with Run().context(RunConfig(nranks=1, experiment=args.experiment)):
        config = ColBERTConfig(
            root=args.root,
            bsize=args.bsize,
            lr=args.lr,
            warmup=args.warmup,
            doc_maxlen=args.doc_maxlen,
            dim=args.dim,
            nway=args.nway,
            accumsteps=args.accumsteps,
            similarity="cosine",
            use_ib_negatives=True,
        )
        trainer = Trainer(
            triples=args.triples,
            queries=args.queries,
            collection=args.collection,
            config=config,
        )
        ckpt = trainer.train(checkpoint=args.checkpoint)
        print("checkpoint:", ckpt)


def index_search(args):
    with Run().context(RunConfig(nranks=1, experiment=args.experiment)):
        config = ColBERTConfig(root=args.root, nbits=args.nbits)
        indexer = Indexer(checkpoint=args.checkpoint, config=config)
        indexer.index(name=args.index_name, collection=args.collection)

        searcher = Searcher(index=args.index_name, config=ColBERTConfig(root=args.root))
        queries = Queries(args.queries)
        ranking = searcher.search_all(queries, k=args.k)
        ranking.save(args.ranking_out)
        print("saved:", args.ranking_out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--root", required=True)
    tr.add_argument("--experiment", default="colbert_exp")
    tr.add_argument("--triples", required=True)
    tr.add_argument("--queries", required=True)
    tr.add_argument("--collection", required=True)
    tr.add_argument("--checkpoint", default="colbert-ir/colbertv2.0")
    tr.add_argument("--bsize", type=int, default=16)
    tr.add_argument("--lr", type=float, default=1e-5)
    tr.add_argument("--warmup", type=int, default=500)
    tr.add_argument("--doc_maxlen", type=int, default=180)
    tr.add_argument("--dim", type=int, default=128)
    tr.add_argument("--nway", type=int, default=16)
    tr.add_argument("--accumsteps", type=int, default=1)

    se = sub.add_parser("index_search")
    se.add_argument("--root", required=True)
    se.add_argument("--experiment", default="colbert_exp")
    se.add_argument("--checkpoint", required=True)
    se.add_argument("--collection", required=True)
    se.add_argument("--queries", required=True)
    se.add_argument("--index_name", default="demo.nbits=2")
    se.add_argument("--nbits", type=int, default=2)
    se.add_argument("--k", type=int, default=100)
    se.add_argument("--ranking_out", required=True)

    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    else:
        index_search(args)


if __name__ == "__main__":
    main()
