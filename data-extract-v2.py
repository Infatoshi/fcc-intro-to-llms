import os
import lzma
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import concurrent.futures

def process_file(args):
    directory, filename, output_file, vocab = args
    file_path = os.path.join(directory, filename)
    with lzma.open(file_path, "rt", encoding="utf-8") as infile:
        text = infile.read()
    with open(output_file, "a", encoding="utf-8") as outfile:
        outfile.write(text)
    characters = set(text)
    return characters

def xz_files_in_dir(directory):
    return [filename for filename in os.listdir(directory) if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename))]

def process_files_in_parallel(files, folder_path, output_file):
    vocab = set()
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        args = [(folder_path, filename, output_file, vocab) for filename in files]
        for characters in tqdm(executor.map(process_file, args), total=len(files)):
            vocab.update(characters)
    return vocab

folder_path = "openwebtext"
output_file_train = "output_train.txt"
output_file_val = "output_val.txt"
vocab_file = "vocab.txt"

files = xz_files_in_dir(folder_path)
total_files = len(files)

split_index = int(total_files * 0.9)  # 90% for training
files_train = files[:split_index]
files_val = files[split_index:]

# Ensure output files are empty before appending
open(output_file_train, 'w').close()
open(output_file_val, 'w').close()

# Process the training files
vocab_train = process_files_in_parallel(files_train, folder_path, output_file_train)

# Process the validation files
vocab_val = process_files_in_parallel(files_val, folder_path, output_file_val)

# Combine vocabularies (if needed) and write to vocab.txt
vocab = vocab_train.union(vocab_val)
with open(vocab_file, "w", encoding="utf-8") as vfile:
    for char in sorted(vocab):
        vfile.write(char + '\n')
