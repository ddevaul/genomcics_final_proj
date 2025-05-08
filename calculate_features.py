import pandas as pd
import re
import time
import multiprocessing as mp
from tqdm import tqdm
import itertools
import gc
from collections import Counter

def calculate_gapped_dinucleotides(df):
    """Calculate gapped dinucleotide frequencies for k=1,2,3."""
    k_values = [1, 2, 3]
    nucleotides = ['A', 'C', 'G', 'T']
    dinucleotides = [a + b for a in nucleotides for b in nucleotides]
    
    # Initialize columns
    for k in k_values:
        for dinuc in dinucleotides:
            df[f"{dinuc}_k{k}"] = 0.0
    
    # Calculate frequencies
    for idx, row in df.iterrows():
        seq = row['sequence']
        for k in k_values:
            counts = Counter()
            total = 0
            
            for i in range(len(seq) - (k + 1)):
                first = seq[i]
                second = seq[i + k + 1]
                if first in nucleotides and second in nucleotides:
                    counts[first + second] += 1
                    total += 1
            
            for dinuc in dinucleotides:
                freq = counts[dinuc] / total if total > 0 else 0
                df.at[idx, f"{dinuc}_k{k}"] = freq
    
    return df

# --- Load the Excel file ---
# Note: You'll need to add code to load your file here
# For example: df = pd.read_excel("your_file.xlsx")

# Remove or comment out this section since it's now handled in process_sequence_file()
# --- Parameters ---
# k_values = [1, 2, 3]  # Calculate for k=1, k=2, and k=3
# nucleotides = ['A', 'C', 'G', 'T']
# dinucleotides = [a + b for a in nucleotides for b in nucleotides]

# --- Ensure DNA seq column is upper-case strings ---
# df['sequence'] = df['sequence'].astype(str).str.upper()

# --- Initialize empty columns for each dinucleotide and k value ---
# for k in k_values:
#     for dinuc in dinucleotides:
#         df[f"{dinuc}_k{k}"] = 0.0

# --- Loop through each row and compute gapped dinucleotide frequencies for each k ---
# for idx, row in df.iterrows():
#     seq = row['sequence']
#     
#     for k in k_values:
#         counts = Counter()
#         total = 0
# 
#         for i in range(len(seq) - k - 1):
#             first = seq[i]
#             second = seq[i + k + 1]
#             if first in nucleotides and second in nucleotides:
#                 dinuc = first + second
#                 counts[dinuc] += 1
#                 total += 1
# 
#         for dinuc in dinucleotides:
#             freq = counts[dinuc] / total if total > 0 else 0
#             df.at[idx, f"{dinuc}_k{k}"] = freq
# 
# # Change the file name as needed
# df.to_csv(f"{file_name}_gapped_dinucleotide_k123.tsv", sep='\t', index=False)


import pandas as pd
import re
import time
import multiprocessing as mp
from tqdm import tqdm
import itertools
import gc
import os

def load_motifs(motif_file):
    """Load motifs from a text file."""
    with open(motif_file, 'r') as f:
        # One motif per line
        motifs = [line.strip() for line in f if line.strip()]
    return motifs

def compile_patterns(motifs):
    """Compile regex patterns for each motif."""
    patterns = {}
    for motif in motifs:
        # Create regex pattern that properly handles N wildcards
        pattern = motif.replace('N', '[ACGT]')
        patterns[motif] = re.compile(pattern)
    return patterns

def preprocess_sequence(sequence):
    """
    Preprocess DNA sequence:
    - Replace stretches of >3 consecutive Ns with a marker
    - Keep 1-3 consecutive Ns (they'll be handled separately)
    """
    # Find stretches of more than 3 consecutive Ns and replace them
    return re.sub('N{4,}', 'X' * 4, sequence)

def generate_sequence_variations(sequence):
    """
    Generate all possible variations of a sequence by replacing N with each nucleotide.
    Only handles sequences with a reasonable number of Ns to avoid combinatorial explosion.
    """
    # Count number of Ns
    n_count = sequence.count('N')
    
    # If too many Ns, return the original sequence
    # (this is a safety check to avoid memory issues)
    if n_count > 5:  # Reduced from 10 to 5 to limit memory usage
        return [sequence]
    
    # No Ns, return original
    if n_count == 0:
        return [sequence]
    
    # Generate all possible combinations of ACGT for each N
    variations = []
    nucleotides = ['A', 'C', 'G', 'T']
    
    # Find positions of all Ns
    n_positions = [i for i, char in enumerate(sequence) if char == 'N']
    
    # Generate all combinations of nucleotides for the N positions
    for combo in itertools.product(nucleotides, repeat=n_count):
        # Create a new sequence with Ns replaced
        new_seq = list(sequence)
        for pos, nucleotide in zip(n_positions, combo):
            new_seq[pos] = nucleotide
        variations.append(''.join(new_seq))
    
    return variations

def search_sequence(args):
    """Search a single sequence for all motifs and return binary indicators."""
    sequence_id, sequence, patterns = args
    results = {}
    
    # Preprocess the sequence to handle long stretches of Ns
    processed_sequence = preprocess_sequence(sequence)
    
    for motif, motif_pattern in patterns.items():
        match_found = False
        
        # APPROACH 1: Direct string comparison with wildcard handling
        if len(motif) == len(processed_sequence):
            position_match = True
            for m_char, s_char in zip(motif, processed_sequence):
                if m_char != 'N' and s_char != 'N' and m_char != s_char:
                    position_match = False
                    break
            
            if position_match:
                match_found = True
        
        # APPROACH 2: Generate variations and use regex
        if not match_found:
            if 'N' in processed_sequence and processed_sequence.count('N') <= 5:  # Reduced from 10 to 5
                variations = generate_sequence_variations(processed_sequence)
                for variant in variations:
                    if motif_pattern.search(variant):
                        match_found = True
                        break
                del variations  # Explicitly delete variations to free memory
            elif 'N' not in processed_sequence:
                match_found = motif_pattern.search(processed_sequence) is not None
            else:
                import random
                for _ in range(min(50, 4**processed_sequence.count('N'))):  # Reduced from 100 to 50
                    variant = ''.join(random.choice(['A','C','G','T']) if c == 'N' else c 
                                    for c in processed_sequence)
                    if motif_pattern.search(variant):
                        match_found = True
                        break
        
        results[motif] = 1 if match_found else 0
    
    return sequence_id, results

def process_batch(batch_data, patterns, num_processes=None):
    """Process a batch of sequences using multiprocessing."""
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 4)  # Limit to max 4 processes
    
    args = [(idx, seq, patterns) for idx, seq in batch_data]
    
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(search_sequence, args), total=len(batch_data)))
    
    return dict(results)

def main(tsv_file, motif_file, output_file, batch_size=500):  # Reduced batch size from 1000 to 500
    start_time = time.time()
    
    # Load motifs
    motifs = load_motifs(motif_file)
    print(f"Loaded {len(motifs)} motifs")
    patterns = compile_patterns(motifs)
    
    # Count degenerate motifs
    degenerate_count = sum(1 for motif in motifs if 'N' in motif)
    print(f"Found {degenerate_count} degenerate motifs (containing N)")
    
    # Load the TSV file using pandas
    print(f"Reading sequence data from {tsv_file}")
    df = pd.read_csv(tsv_file, sep='\t')
    total_sequences = len(df)
    print(f"Processing {total_sequences} sequences")
    
    # Process sequences in batches
    all_results = {}
    
    # Create batches of sequences
    for i in range(0, total_sequences, batch_size):
        batch_df = df.iloc[i:min(i+batch_size, total_sequences)]
        
        # Create a list of (sequence_id, sequence) tuples
        batch_data = [(f"{row['chrom']}:{row['start']}-{row['end']}", row['sequence']) 
                      for _, row in batch_df.iterrows()]
        
        print(f"Processing batch {i//batch_size + 1}/{(total_sequences+batch_size-1)//batch_size}")
        batch_results = process_batch(batch_data, patterns)
        
        # Merge results
        all_results.update(batch_results)
        
        # Clear memory after each batch
        del batch_df
        del batch_data
        del batch_results
        gc.collect()
    
    # Create a DataFrame from the results
    print("Creating results dataframe")
    result_rows = []
    for seq_id, motif_results in all_results.items():
        row = {'sequence_id': seq_id}
        row.update(motif_results)
        result_rows.append(row)
    
    result_df = pd.DataFrame(result_rows)
    
    # Extract chromosome, start, and end from sequence_id
    result_df[['chrom', 'position']] = result_df['sequence_id'].str.split(':', expand=True)
    result_df[['start', 'end']] = result_df['position'].str.split('-', expand=True)
    result_df = result_df.drop('position', axis=1)
    
    # Merge with original data
    df['sequence_id'] = df['chrom'] + ':' + df['start'].astype(str) + '-' + df['end'].astype(str)
    merged_df = pd.merge(df, result_df.drop(['chrom', 'start', 'end'], axis=1), on='sequence_id', how='left')
    
    # Save results
    print(f"Writing results to {output_file}")
    for motif in motifs:
        if motif in merged_df.columns:
            merged_df[motif] = merged_df[motif].fillna(0).astype(int)
    
    # Save to TSV
    merged_df.to_csv(output_file, sep='\t', index=False)
    
    end_time = time.time()
    print(f"Processed {total_sequences} sequences in {end_time - start_time:.2f} seconds")
    print(f"Results written to {output_file}")

def process_sequence_file(file_name):
    """Process a single sequence file for both gapped dinucleotides and motifs."""
    print(f"Processing file: {file_name}")
    
    # Read input file
    df = pd.read_csv(f'{file_name}.tsv', sep='\t', engine='python')
    df['sequence'] = df['sequence'].astype(str).str.upper()
    
    # Step 1: Calculate gapped dinucleotides
    print("Calculating gapped dinucleotides...")
    df = calculate_gapped_dinucleotides(df)
    
    # Save with dinucleotides
    intermediate_file = f"{file_name}_with_dinucleotides.tsv"
    df.to_csv(intermediate_file, sep='\t', index=False)
    print(f"Saved dinucleotide features to {intermediate_file}")
    
    # Step 2: Process motifs using the file we just saved
    print("Processing motifs...")
    main(intermediate_file, "motifs.txt", f"{file_name}_with_dinucleotides_and_motifs.tsv")

if __name__ == "__main__":
    # Single entry point for processing
    file_name = input("Enter the file name (without .tsv extension): ")
    process_sequence_file(file_name)

    