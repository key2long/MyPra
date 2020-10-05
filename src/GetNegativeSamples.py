# -*- coding: utf-8 -*-
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--neg_samp_size', default=100, type=int)
    args = parser.parse_args()
    # args.graph_names = ["fb15k237", "WN18RR", "YAGO3_10"]
    args.graph_names = ["fb15k237"]
    # args.processed_path = Path("../DATA/processed")
    args.processed_path = Path("../DATA/processed/fb15k237/test_case")  # 做测试代码时使用
    args.hold_out_num = 10

if __name__ == "__main__":
    main()
