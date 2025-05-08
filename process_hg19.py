#!/usr/bin/env python3
import pandas as pd
import numpy as np
import random
from twobitreader import TwoBitFile
from tqdm import tqdm
import csv
import argparse
from multiprocessing.pool import ThreadPool
from functools import partial

def get_recomb_rate(chrom, pos, recomb_df):
    # Special case for chromosome Y, which has no recombination data
    if chrom == 'chrY':
        return 0.0
        
    # Find all rows for this chromosome
    chrom_recomb = recomb_df[recomb_df['chrom'] == chrom]
    if len(chrom_recomb) == 0:
        return None
    
    # Find the row where position falls between chromStart and chromEnd
    matching_rows = chrom_recomb[(chrom_recomb['chromStart'] <= pos) & 
                                (chrom_recomb['chromEnd'] > pos)]
    
    if len(matching_rows) > 0:
        return matching_rows.iloc[0]['decodeAvg']
    else:
        if pos < chrom_recomb['chromStart'].min():
            closest_row = chrom_recomb.loc[chrom_recomb['chromStart'].idxmin()]
            return closest_row['decodeAvg']
        elif pos >= chrom_recomb['chromEnd'].max():
            closest_row = chrom_recomb.loc[chrom_recomb['chromEnd'].idxmax()]
            return closest_row['decodeAvg']
        else:
            return None

def check_overlap(new_start, new_end, existing_regions):
    """Check if a new region overlaps with any existing regions"""
    for _, region in existing_regions.iterrows():
        if (new_start < region['end'] and new_end > region['start']):
            return True
    return False

def get_random_sequence(genome, chrom, length, existing_regions, pbar=None):
    """Get a random sequence that doesn't overlap with existing regions"""
    # Get chromosome length
    chrom_length = len(genome[chrom])
    
    # Filter existing regions for the current chromosome
    chrom_regions = existing_regions[existing_regions['chrom'] == chrom]
    
    if len(chrom_regions) == 0:
        # If no existing regions, just return a random sequence
        start = random.randint(0, chrom_length - length)
        return genome[chrom][start:start + length].upper(), start, start + length
    
    # Convert regions to numpy arrays for faster processing
    starts = chrom_regions['start'].values
    ends = chrom_regions['end'].values
    
    # Generate multiple random positions at once
    n_attempts = 100
    positions = np.random.randint(0, chrom_length - length, size=n_attempts)
    
    # Check all positions at once using numpy operations
    for start in positions:
        end = start + length
        # Check if this position overlaps with any region
        if not np.any((start < ends) & (end > starts)):
            return genome[chrom][start:end].upper(), start, end
    
    return None, None, None

def combine_sequences(hotspot_path, twobit_path, output_path):
    """Step 1: Combine sequences with hotspots"""
    print("\nStep 1: Combining sequences with hotspots...")
    genome = TwoBitFile(twobit_path)

    with open(hotspot_path, 'r') as fin, open(output_path, 'w', newline='') as fout:
        # skip leading ## comments
        while True:
            pos = fin.tell()
            line = fin.readline()
            if not line.startswith('#'):
                fin.seek(pos)
                break

        reader = csv.reader(fin, delimiter='\t')
        cols = next(reader)                           # header row
        writer = csv.writer(fout)
        writer.writerow(cols + ['sequence'])          # add seq column

        n_written = 0
        for row in reader:
            chrom, start_s, end_s = row[0], row[1], row[2]
            start, end = int(start_s), int(end_s)

            # 1) convert to 0-based half-open
            zb_start = start - 1
            zb_end   = end

            # 2) clip to contig bounds
            contig_len = len(genome[chrom])
            zb_start = max(0, zb_start)
            zb_end   = min(contig_len, zb_end)

            # 3) skip any invalid intervals
            if zb_end <= zb_start:
                print(f"[WARN] skipping invalid region {chrom}:{start}-{end}")
                seq = ''
            else:
                seq = genome[chrom][zb_start:zb_end]

            writer.writerow(row + [seq])
            n_written += 1

    print(f"Wrote {n_written} hotspots (with sequences) to {output_path}")
    return output_path

def add_recombination_rates(input_path, output_path):
    """Step 2: Add recombination rates"""
    print("\nStep 2: Adding recombination rates...")
    
    # Read in your hotspot table with the correct separator
    # First try to read and split the columns properly
    hot = pd.read_csv(input_path, sep='\t')
    
    # If the columns are still comma-separated in a single column, fix them
    if len(hot.columns) == 1:
        # Split the single column into multiple columns
        hot = pd.read_csv(input_path)
    
    print("Input columns after fixing:", hot.columns.tolist())

    # Read in the recombination-rate table
    recomb = pd.read_csv('hg19recombRate.txt', sep='\t', comment='#')
    recomb.columns = ['chrom', 'chromStart', 'chromEnd', 'name', 'decodeAvg', 'decodeFemale', 
                     'decodeMale', 'marshfieldAvg', 'marshfieldFemale', 'marshfieldMale',
                     'genethonAvg', 'genethonFemale', 'genethonMale']

    # Apply the function to each hotspot
    hot['recomb_rate'] = hot.apply(
        lambda row: get_recomb_rate(row['chrom'], row['start'], recomb), 
        axis=1
    )

    # Save the result with tab separator
    hot.to_csv(output_path, sep='\t', index=False)

    # Print summary statistics
    total_hotspots = len(hot)
    missing_rates = hot['recomb_rate'].isna().sum()
    chrY_hotspots = len(hot[hot['chrom'] == 'chrY'])
    chrY_with_rates = len(hot[(hot['chrom'] == 'chrY') & hot['recomb_rate'].notna()])

    print("Done! Each hotspot row now has a 'recomb_rate' field.")
    print(f"Hotspots with recombination rates: {hot['recomb_rate'].notna().sum()} out of {total_hotspots} ({hot['recomb_rate'].notna().sum()/total_hotspots*100:.2f}%)")
    print(f"Chromosome Y hotspots with rates: {chrY_with_rates} out of {chrY_hotspots} ({chrY_with_rates/chrY_hotspots*100 if chrY_hotspots > 0 else 0:.2f}%)")
    return output_path

def create_negative_dataset(input_path, output_path):
    """Step 3: Create negative dataset"""
    print("\nStep 3: Creating negative dataset...")
    # Read the original dataset
    df = pd.read_csv(input_path, sep='\t')
    
    # Read recombination rates
    recomb = pd.read_csv('hg19recombRate.txt', sep='\t', comment='#')
    recomb.columns = ['chrom', 'chromStart', 'chromEnd', 'name', 'decodeAvg', 'decodeFemale', 
                     'decodeMale', 'marshfieldAvg', 'marshfieldFemale', 'marshfieldMale',
                     'genethonAvg', 'genethonFemale', 'genethonMale']
    
    # Load the genome
    genome = TwoBitFile('hg19')
    
    # Create a new list for negative samples
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
    
    # Convert to dataframe
    negative_df = pd.DataFrame(negative_samples)
    
    # Combine with original dataset
    combined_df = pd.concat([df, negative_df], ignore_index=True)
    
    # Calculate means
    combined_df['AA_mean'] = combined_df[['AA1_strength', 'AA2_strength']].mean(axis=1)
    combined_df['AB_mean'] = combined_df[['AB1_strength', 'AB2_strength']].mean(axis=1)
    combined_df['A_mean'] = combined_df[['AA_mean', 'AB_mean']].mean(axis=1)
    combined_df['strength'] = combined_df[['A_mean', 'AC_strength']].mean(axis=1)
    
    # Normalize strength values
    combined_df['strength'] = (combined_df['strength'] - combined_df['strength'].mean()) / combined_df['strength'].std()
    
    # Keep only the desired columns
    final_df = combined_df[['chrom', 'start', 'end', 'strength', 'sequence', 'recomb_rate']].copy()
    
    # Convert sequences to uppercase
    final_df['sequence'] = final_df['sequence'].str.upper()
    
    # Exclude chrX and chrY
    final_df = final_df[final_df['chrom'] != 'chrX']
    final_df = final_df[final_df['chrom'] != 'chrY']
    
    # Save to output file
    final_df.to_csv(output_path, sep='\t', index=False)
    
    print(f"Added {len(negative_samples)} negative samples to the dataset")
    return output_path

def cleanup_dataset(input_path):
    """Step 4: Clean up dataset and create various versions"""
    print("\nStep 4: Cleaning up dataset and creating versions...")
    # Read the dataset
    df = pd.read_csv(input_path, sep='\t')
    
    # Count initial rows
    initial_rows = len(df)
    
    # Fill empty strength values with 0
    df['strength'] = df['strength'].fillna(0)
    
    # Remove rows with no recombination rate
    df = df.dropna(subset=['recomb_rate'])
    
    # Shift strength values for non-zero entries (starting at 1)
    min_strength = df[df['strength'] != 0]['strength'].min()
    df.loc[df['strength'] != 0, 'strength'] = df[df['strength'] != 0]['strength'] - min_strength
    
    # Create binary recombination rate version
    mean_recomb = df['recomb_rate'].mean()
    binary_df = df.copy()
    binary_df['recomb_rate'] = (binary_df['recomb_rate'] > mean_recomb).astype(int)
    
    # Create thresholded versions
    three_hot_df = df.copy()
    three_hot_df['recomb_rate'] = (three_hot_df['recomb_rate'] > 3).astype(int)
    
    four_hot_df = df.copy()
    four_hot_df['recomb_rate'] = (four_hot_df['recomb_rate'] > 4).astype(int)
    
    five_hot_df = df.copy()
    five_hot_df['recomb_rate'] = (five_hot_df['recomb_rate'] > 5).astype(int)
    
    # Save all versions
    df.to_csv('processed_hg19_with_negatives_cleaned.tsv', sep='\t', index=False)
    binary_df.to_csv('processed_hg19_with_negatives_binary.tsv', sep='\t', index=False)
    three_hot_df.to_csv('processed_hg19_with_negatives_3hot.tsv', sep='\t', index=False)
    four_hot_df.to_csv('processed_hg19_with_negatives_4hot.tsv', sep='\t', index=False)
    five_hot_df.to_csv('processed_hg19_with_negatives_5hot.tsv', sep='\t', index=False)
    
    # Print summary
    print("\nCleaning complete!")
    print(f"Initial number of rows: {initial_rows}")
    print(f"Final number of rows: {len(df)}")
    print(f"Rows removed: {initial_rows - len(df)}")
    
    print("\nStrength values (all versions):")
    print(f"Rows with strength = 0: {len(df[df['strength'] == 0])}")
    print(f"Rows with strength > 0: {len(df[df['strength'] > 0])}")
    print(f"Minimum strength value: {df['strength'].min():.3f}")
    print(f"Maximum strength value: {df['strength'].max():.3f}")
    
    print("\nRecombination rate statistics:")
    print(f"Mean recombination rate: {mean_recomb:.3f}")
    print(f"Rows with recombination rate > 3: {len(df[df['recomb_rate'] > 3])} ({len(df[df['recomb_rate'] > 3])/len(df)*100:.2f}%)")
    print(f"Rows with recombination rate > 4: {len(df[df['recomb_rate'] > 4])} ({len(df[df['recomb_rate'] > 4])/len(df)*100:.2f}%)")
    print(f"Rows with recombination rate > 5: {len(df[df['recomb_rate'] > 5])} ({len(df[df['recomb_rate'] > 5])/len(df)*100:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description="Process hotspot data through the complete pipeline")
    parser.add_argument('-i', '--input', required=True, help="Input hotspot file (TSV)")
    parser.add_argument('-g', '--genome', required=True, help="Genome file in .2bit format")
    args = parser.parse_args()

    # Step 1: Combine sequences with hotspots
    combined_path = combine_sequences(args.input, args.genome, 'combined_with_sequences.tsv')
    
    # Step 2: Add recombination rates
    recomb_path = add_recombination_rates(combined_path, 'combined_recomb_hg19.tsv')
    
    # Step 3: Create negative dataset
    negative_path = create_negative_dataset(recomb_path, 'processed_hg19_with_negatives.tsv')
    
    # Step 4: Clean up dataset
    cleanup_dataset(negative_path)

if __name__ == "__main__":
    main() 