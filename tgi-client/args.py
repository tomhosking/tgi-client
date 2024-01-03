import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="TGI Client runner",
    )

    # Paths
    parser.add_argument("-d", "--data_path", type=str, metavar="DATA", default="./data/", help="Path to data sources")
    parser.add_argument(
        "-o", "--output_path", type=str, metavar="OUTPUT", default="./runs/", help="Path to output folder"
    )

    parser.add_argument(
       "--endpoint", type=str, metavar="ENDPOINT", default="http://127.0.0.1:8080", help="Endpoint address"
    )
    
    args = parser.parse_args()

    return args
