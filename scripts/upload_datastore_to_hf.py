import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Upload local MLX datastore to HuggingFace Hub")
    parser.add_argument("repo_id", type=str, help="Your HuggingFace repo ID (e.g., 'your-username/mlx_datastore')")
    parser.add_argument(
        "--local-dir",
        type=str,
        default="/Users/andrew/DataDump/datasets/mlx_datastore",
        help="Path to the local datastore directory",
    )
    parser.add_argument("--private", default=False, action="store_true", help="Make the HuggingFace repository private")
    args = parser.parse_args()

    local_path = Path(args.local_dir)
    if not local_path.exists():
        print(f"Error: Local directory {local_path} does not exist.")
        return

    api = HfApi()

    print(f"Creating dataset repository '{args.repo_id}' (private={args.private})...")
    try:
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    except Exception as e:
        print(f"Note: Repo creation skipped or failed (might already exist): {e}")

    print(f"Uploading contents of {local_path} to {args.repo_id}...")
    print("This may take a while depending on your internet connection and dataset size.")

    # upload_large_folder automatically handles large files, parallel uploads, and resuming
    api.upload_large_folder(
        folder_path=str(local_path),
        repo_id=args.repo_id,
        repo_type="dataset",
        num_workers=8,
        ignore_patterns=[".DS_Store"],  # Ignore macOS system files
        print_report=True,
        print_report_every=10,
        private=args.private,
    )

    print(f"\nSuccess! Dataset uploaded to: https://huggingface.co/datasets/{args.repo_id}")
    print(f"You can now train using: --cache-hf {args.repo_id}")


if __name__ == "__main__":
    main()
