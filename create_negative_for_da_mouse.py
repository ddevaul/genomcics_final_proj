import pandas as pd
import numpy as np
import random
from twobitreader import TwoBitFile
from tqdm import tqdm

def get_recomb_rate(chrom, pos, recomb_df):
    # Convert position from base pairs to kilobases
    pos_kb = pos / 1000
    
    # Find all rows for this chromosome
    chrom_recomb = recomb_df[recomb_df['Chr'] == chrom]
    if len(chrom_recomb) == 0:
        return None
    
    # Find the closest position
    distances = abs(chrom_recomb['Kb'] - pos_kb)
    closest_idx = distances.idxmin()
    closest_row = chrom_recomb.loc[closest_idx]
    
    # Convert 4Ner/kb to cM/Mb (multiply by 1000)
    return closest_row['4Ner/kb'] * 1000

def check_overlap(new_start, new_end, existing_regions):
    """Check if a new region overlaps with any existing regions"""
    for _, region in existing_regions.iterrows():
        if (new_start < region['end'] and new_end > region['start']):
            return True
    return False

def get_random_sequence(genome, chrom, length, existing_regions, pbar=None):
    # Get chromosome length
    chrom_length = len(genome[chrom])
    
    # Filter existing regions for the current chromosome
    chrom_regions = existing_regions[existing_regions['chrom'] == chrom]
    
    # Keep trying until we find a non-overlapping region
    max_attempts = 1000  # Prevent infinite loops
    attempts = 0
    
    # Create a progress bar for attempts if no parent progress bar is provided
    if pbar is None:
        attempt_pbar = tqdm(total=max_attempts, desc=f"Finding non-overlapping region for {chrom}", leave=False)
    else:
        attempt_pbar = None
    
    while attempts < max_attempts:
        # Generate random start position
        start = random.randint(0, chrom_length - length)
        end = start + length
        
        # Check if it overlaps with any existing regions
        if not check_overlap(start, end, chrom_regions):
            # If no overlap, return the sequence
            if attempt_pbar is not None:
                attempt_pbar.close()
            return genome[chrom][start:end].upper(), start, end
        
        attempts += 1
        if attempt_pbar is not None:
            attempt_pbar.update(1)
    
    if attempt_pbar is not None:
        attempt_pbar.close()
    print(f"Warning: Could not find non-overlapping region for chromosome {chrom} after {max_attempts} attempts")
    return None, None, None

def main():
    # Read the original dataset
    print("Reading input files...")
    df = pd.read_csv('processed_mouse.tsv', sep='\t')
    
    # Read recombination rates
    print("Reading recombination rates...")
    recomb = pd.read_csv('recomb_rate.tsv', sep='\t')
    
    # Load the genome
    print("Loading genome...")
    genome = TwoBitFile('mm10.2bit')
    
    # Create a new dataframe for negative samples
    negative_samples = []
    
    # For each sequence in the original dataset
    print("Generating negative samples...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing sequences"):
        # Get a random sequence of the same length
        sequence, start, end = get_random_sequence(genome, row['chrom'], row['end'] - row['start'], df)
        
        if sequence is not None:
            # Get recombination rate for the new position
            recomb_rate = get_recomb_rate(row['chrom'], start, recomb)
            
            # Add to negative samples
            negative_samples.append({
                'chrom': row['chrom'],
                'start': start,
                'end': end,
                'strength': 0,  # Negative samples have strength 0
                'sequence': sequence,
                'recomb_rate': recomb_rate
            })
    
    print("Processing results...")
    # Convert to dataframe
    negative_df = pd.DataFrame(negative_samples)
    
    # Combine with original dataset
    combined_df = pd.concat([df, negative_df], ignore_index=True)
    
    # Normalize strength values
    combined_df['strength'] = (combined_df['strength'] - combined_df['strength'].mean()) / combined_df['strength'].std()
    
    # Keep only the desired columns
    final_df = combined_df[['chrom', 'start', 'end', 'strength', 'sequence', 'recomb_rate']].copy()
    
    # Convert sequences to uppercase
    final_df['sequence'] = final_df['sequence'].str.upper()
    
    # Save to a new TSV file
    print("Saving results...")
    final_df.to_csv('processed_mouse_with_negatives.tsv', sep='\t', index=False)
    
    print("\nProcessing complete! Output saved to processed_mouse_with_negatives.tsv")
    print(f"Strength values have been normalized to range: {final_df['strength'].min():.3f} to {final_df['strength'].max():.3f}")
    print(f"Added {len(negative_samples)} negative samples to the dataset")

if __name__ == "__main__":
    main()
