import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="TGI Client runner",
    )

    # Paths
    parser.add_argument("-d", "--data_path", type=str, metavar="DATA", default="./data/", help="Path to data sources")
    parser.add_argument(
        "-i", "--input", type=str, metavar="INPUT", default="./input.jsonl", help="Path to input file"
    )
    parser.add_argument(
        "-o", "--output", type=str, metavar="OUTPUT", default="./output.jsonl", help="Path to output file"
    )

    parser.add_argument(
       "--endpoint", type=str, metavar="ENDPOINT", default="http://127.0.0.1:8080", help="Endpoint address"
    )
    parser.add_argument(
       "--model", type=str, metavar="MODEL", default="mistralai/Mistral-7B-Instruct-v0.2", help="HF model slug to use for formatting"
    )

    
    
    args = parser.parse_args()

    return args
