import argparse

from qdrant import get_store

BGE_PREFIX = "Represent this sentence for searching relevant passages: "


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense search over the collection")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5, help="number of results")
    parser.add_argument("--prefix", action="store_true", help="add the BGE query prefix")
    args = parser.parse_args()

    query = BGE_PREFIX + args.query if args.prefix else args.query

    store = get_store()  # dense mode by default
    results = store.similarity_search_with_score(query, k=args.k)

    for rank, (doc, score) in enumerate(results, start=1):
        snippet = doc.page_content[:200].replace("\n", " ")
        print(f"\n#{rank}  score={score:.3f}  {doc.metadata['filename']}  p.{doc.metadata['page']}")
        print(f"    {snippet}...")


if __name__ == "__main__":
    main()