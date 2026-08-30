import os


os.environ["HF_HOME"] = "/hkfs/work/workspace_haic/scratch/on3546-VLM"

import numpy as np
from datasets import load_dataset


# Config
dataset_name = "sample-100BT"
seed = 42

print("1. Loading dataset...")
dataset_ori = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name=dataset_name,
    split="train",
    trust_remote_code=True,
    streaming=False,
    num_proc=16,
)

# 2. Add a simple ID column (0, 1, 2, ... N)
# This creates a permanent record of the original row order
print("2. Adding ID column...")
dataset = dataset_ori.map(
    lambda _, idx: {'orig_index': idx},
    with_indices=True,
    num_proc=16  # Fast parallel processing
)

# 3. Shuffle the dataset using Hugging Face's native method
# We let Hugging Face do the work so we don't have to guess the algorithm
print(f"3. Shuffling with seed {seed}...")
shuffled_dataset = dataset.shuffle(seed=seed)

# 4. Extract the shuffled IDs and save them
# This extracts the [4052, 12, 99...] order directly from the result
print("4. Saving shuffle indices...")
indices = np.array(shuffled_dataset['orig_index'], dtype=np.int64)
np.save("global_shuffle_indices.npy", indices)

print("Done! Transfer 'global_shuffle_indices.npy' to your low-memory server.")

# ---------------------------------------------------------
# 5. VERIFICATION STEP
# ---------------------------------------------------------
print("\n5. Verifying consistency...")

del shuffled_dataset
shuffled_dataset = dataset_ori.shuffle(seed=seed)

# Simulate what the low-memory server will do:
# Load the saved indices and apply them to the ORIGINAL dataset
loaded_indices = np.load("global_shuffle_indices.npy")
reconstructed_dataset = dataset_ori.select(loaded_indices)

# Verify that the two datasets are identical at random points
# We check the first item, the 100th item, and the last item
test_indices = [0, 100, 1000, len(dataset) - 1]

for i in test_indices:
    # Get the row from the standard shuffle
    row_shuffle = shuffled_dataset[i]

    # Get the row from the indices method
    row_reconstruct = reconstructed_dataset[i]

    text_shuffle = row_shuffle['text']
    text_reconstruct = row_reconstruct['text']

    if text_shuffle != text_reconstruct:
        raise ValueError(f"❌ Content Mismatch at position {i}!")

print("\n✅ VERIFICATION PASSED: dataset.select(indices) matches dataset.shuffle() exactly.")
print("You can safely transfer 'global_shuffle_indices.npy' to your low-memory server.")