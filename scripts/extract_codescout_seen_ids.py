"""Extract instance IDs from released CodeScout training rollouts.

Example:
    python scripts/extract_codescout_seen_ids.py \
        --subset CodeScout_4B \
        --output /root/autodl-tmp/codescout4b_seen_ids.txt
"""

import argparse

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="OpenHands/CodeScout_Training_Rollouts",
        help="Hugging Face dataset containing released CodeScout rollouts.",
    )
    parser.add_argument(
        "--subset",
        default="CodeScout_4B",
        help="Dataset subset/config to read.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to read.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output text file with one instance_id per line.",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, args.subset, split=args.split, streaming=True)
    instance_ids = sorted({row["instance_id"] for row in dataset})

    with open(args.output, "w", encoding="utf-8") as f:
        for instance_id in instance_ids:
            f.write(instance_id + "\n")

    print(f"Wrote {len(instance_ids)} unique instance IDs to {args.output}")


if __name__ == "__main__":
    main()
