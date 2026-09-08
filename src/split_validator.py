import os

processed_path = "/home/keerthi-180205/MY-PROJECTS/neurolock/data/processed"

for split in ["train", "val", "test"]:
    split_path = os.path.join(processed_path, split)

    print(f"\n{split.upper()}")

    for identity in os.listdir(split_path):
        identity_path = os.path.join(split_path, identity)
        count = len(os.listdir(identity_path))

        print(identity, count)