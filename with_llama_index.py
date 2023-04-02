import argparse
import os

from dotenv import load_dotenv
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader

load_dotenv(".env")


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        "-q",
        help="Querty string: e.g. '関門トンネルとはどことどこを結ぶトンネルですか？'",
        required=True,
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    documents = SimpleDirectoryReader("data").load_data()
    index = GPTSimpleVectorIndex.from_documents(documents)

    index.save_to_disk("index.json")
    index = GPTSimpleVectorIndex.load_from_disk("index.json")

    print(index.query(args.query))


if __name__ == "__main__":
    main()
