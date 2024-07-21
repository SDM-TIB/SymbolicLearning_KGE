import argparse
import pandas as pd
from ranks import  tail_prediction, head_prediction


def main(args):
    pred = tail_prediction(args.results, args.head, args.relation, args.training)

    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--results', type=str, required=True, help='results of trained model')
    parser.add_argument('--head', type=str, required=True, help='head entity')
    parser.add_argument('--relation', type=str, required=True, help='relation')
    parser.add_argument('--training', type=str, nargs='+', required=True, help='training triples')

    args = parser.parse_args()
    main(args)