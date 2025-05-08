#!/usr/bin/env python3
import sys
import csv
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from pyfaidx import Fasta
import twobitreader

def check_overlap(new_start, new_end, existing_regions):
    """Check if a new region overlaps with any existing regions"""
    for _, region in existing_regions.iterrows():
        if (new_start < region['end'] and new_end > region['start']):
            return True
    return False

def get_random_sequence(genome, chrom, length, existing_regions, pbar=None):
    """Get a random non-overlapping genomic sequence"""
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
        # Generate random start position - stay safely within chromosome bounds
        start = random.randint(0, max(0, chrom_length - length - 10))  # Add a safety margin
        end = start + length
        
        # Check if it overlaps with any existing regions
        if not check_overlap(start, end, chrom_regions):
            # If no overlap, return the sequence
            if attempt_pbar is not None:
                attempt_pbar.close()
            try:
                # Extract sequence from Fasta object
                seq = str(genome[chrom][start:end]).upper()
                return seq, start, end
            except Exception as e:
                print(f"Error extracting sequence for {chrom}:{start}-{end}: {str(e)}")
                return None, None, None
        
        attempts += 1
        if attempt_pbar is not None:
            attempt_pbar.update(1)
    
    if attempt_pbar is not None:
        attempt_pbar.close()
    print(f"Warning: Could not find non-overlapping region for chromosome {chrom} after {max_attempts} attempts")
    return None, None, None

def extract_and_rate(genome, hotspot_file, rates_file, output_tsv):
    """Extract sequences from genome and compute recombination rates"""
    print(f"Extracting sequences and computing recombination rates...")
    
    # Load recombination map
    rates = pd.read_csv(rates_file, sep='\t')
    # Prefix "chr" to match hotspot file
    rates['chrom'] = 'chr' + rates['chr'].astype(str)
    rates_dict = {
        chrom: dict(zip(g['pos'], g['Sex_avg']))
        for chrom, g in rates.groupby('chrom')
    }
    
    # Get chromosome lengths
    chrom_lengths = {chrom: len(genome[chrom]) for chrom in genome.keys()}
    
    print("Chromosome lengths:")
    for chrom, length in sorted(chrom_lengths.items()):
        print(f"  {chrom}: {length:,} bp")

    # Read hotspots and process
    valid_count = 0
    skipped_count = 0
    with open(hotspot_file) as fin, open(output_tsv, 'w', newline='') as fout:
        reader = csv.DictReader(fin, delimiter='\t')
        # Add two new columns
        fieldnames = reader.fieldnames + ['sequence', 'cM_per_Mb']
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()

        for row in tqdm(reader, desc="Processing hotspots"):
            chrom = row['chr']
            start = int(row['from'])
            end = int(row['to'])
            
            # Check if the chromosome exists in our genome
            if chrom not in chrom_lengths:
                skipped_count += 1
                continue
                
            # Check if coordinates are within chromosome bounds
            if start < 0 or end > chrom_lengths[chrom]:
                skipped_count += 1
                continue

            # Extract DNA sequence
            try:
                # Using Fasta object
                seq = str(genome[chrom][start:end]).upper()
            except Exception as e:
                print(f"Error extracting sequence for {chrom}:{start}-{end}: {str(e)}")
                skipped_count += 1
                continue

            # Map each hotspot end to its nearest 1 kb bin
            bs = ((start - 1) // 1000) * 1000 + 1
            be = ((end - 1) // 1000) * 1000 + 1
            cm_s = rates_dict.get(chrom, {}).get(bs)
            cm_e = rates_dict.get(chrom, {}).get(be)
            if cm_s is not None and cm_e is not None and end > start:
                rate = (cm_e - cm_s) / (end - start) * 1e6
            else:
                rate = ''

            row['sequence'] = seq
            row['cM_per_Mb'] = rate
            writer.writerow(row)
            valid_count += 1
    
    print(f"Processed {valid_count} valid hotspots, skipped {skipped_count} invalid hotspots")
    print(f"→ Wrote {output_tsv}")
    return output_tsv

    print(f"→ Wrote {output_tsv}")
    return output_tsv

def process_hotspots(input_tsv, output_tsv):
    """Process and filter hotspot data"""
    print(f"Processing hotspot data...")
    
    # Read the TSV file
    df = pd.read_csv(input_tsv, sep='\t')

    # Filter rows where hsSRwt > 0 and strSRwt >= 0
    filtered_df = df[(df['hsSRwt'] > 0) & (df['strSRwt'] >= 0)]

    # Exclude chrX
    filtered_df = filtered_df[filtered_df['chr'] != 'chrX']

    # Select and rename columns
    result_df = filtered_df[['chr', 'from', 'to', 'strSRwt', 'sequence', 'cM_per_Mb']].rename(
        columns={
            'chr': 'chrom',
            'from': 'start',
            'to': 'end',
            'strSRwt': 'strength',
            'cM_per_Mb': 'recomb_rate'
        }
    )

    # Normalize the strength column using mean/std normalization
    mean_strength = result_df['strength'].mean()
    std_strength = result_df['strength'].std()
    result_df['strength'] = (result_df['strength'] - mean_strength) / std_strength

    # Keep only the desired columns
    result_df = result_df[['chrom', 'start', 'end', 'strength', 'sequence', 'recomb_rate']].copy()

    # Convert sequences to uppercase
    result_df['sequence'] = result_df['sequence'].str.upper()

    # Output to a new file
    result_df.to_csv(output_tsv, sep='\t', index=False)

    print(f"Processing complete. Output saved to '{output_tsv}'")
    print(f"Filtered from {len(df)} to {len(result_df)} rows")
    
    return output_tsv

def generate_negative_samples(genome, processed_tsv, output_tsv):
    """Generate negative samples for the dataset"""
    print(f"Generating negative samples...")
    
    # Read the processed dataset
    df = pd.read_csv(processed_tsv, sep='\t')
    
    # Check if we need to add/remove 'chr' prefix
    genome_has_chr_prefix = any(key.startswith('chr') for key in genome.keys())
    hotspot_has_chr_prefix = df['chrom'].iloc[0].startswith('chr')
    
    print(f"Genome has chr prefix: {genome_has_chr_prefix}, Hotspots have chr prefix: {hotspot_has_chr_prefix}")
    
    if genome_has_chr_prefix and not hotspot_has_chr_prefix:
        print("Adding 'chr' prefix to hotspots")
        df['chrom'] = 'chr' + df['chrom']
    elif not genome_has_chr_prefix and hotspot_has_chr_prefix:
        print("Removing 'chr' prefix from hotspots")
        df['chrom'] = df['chrom'].str.replace('chr', '')
    
    # Create a new dataframe for negative samples
    negative_samples = []
    
    # For each sequence in the original dataset
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing sequences"):
        try:
            # Get a random sequence of the same length
            sequence, start, end = get_random_sequence(genome, row['chrom'], row['end'] - row['start'], df)
            
            if sequence is not None:
                # Add to negative samples with strength 0
                negative_samples.append({
                    'chrom': row['chrom'],
                    'start': start,
                    'end': end,
                    'strength': 0,  # Negative samples have strength 0
                    'sequence': sequence,
                    'recomb_rate': row['recomb_rate']  # Keep the same recombination rate
                })
        except Exception as e:
            print(f"Error generating negative sample for {row['chrom']}:{row['start']}-{row['end']}: {str(e)}")
            continue
    
    # Convert to dataframe
    negative_df = pd.DataFrame(negative_samples)
    
    # Combine with original dataset
    combined_df = pd.concat([df, negative_df], ignore_index=True)
    
    # Save to a new TSV file
    combined_df.to_csv(output_tsv, sep='\t', index=False)
    
    print(f"Added {len(negative_samples)} negative samples to the dataset")
    print(f"Output saved to {output_tsv}")
    
    return output_tsv

def create_thresholded_versions(input_tsv, output_prefix):
    """Create different versions of the dataset with various thresholds"""
    print(f"Creating thresholded versions...")
    
    # Read the dataset
    df = pd.read_csv(input_tsv, sep='\t')
    
    # Count initial rows
    initial_rows = len(df)
    
    # Fill empty strength values with 0
    df['strength'] = df['strength'].fillna(0)
    
    # Remove rows with no recombination rate
    df = df.dropna(subset=['recomb_rate'])
    
    # Shift strength values for non-zero entries (starting at 1)
    # Get the minimum strength value from non-zero entries
    min_strength = df[df['strength'] != 0]['strength'].min()
    
    # For non-zero entries, shift up to make 1 the minimum
    df.loc[df['strength'] != 0, 'strength'] = df[df['strength'] != 0]['strength'] - min_strength + 1
    
    # Create binary recombination rate version
    mean_recomb = df['recomb_rate'].mean()
    binary_df = df.copy()
    binary_df['recomb_rate'] = (binary_df['recomb_rate'] > mean_recomb).astype(int)
    
    # Create thresholded versions
    thresholds = [1.2, 1.5, 2, 3, 4, 5]
    threshold_dfs = {}
    
    for threshold in thresholds:
        thresh_df = df.copy()
        thresh_df['recomb_rate'] = (thresh_df['recomb_rate'] > threshold).astype(int)
        threshold_dfs[threshold] = thresh_df
    
    # Save all versions
    df.to_csv(f"{output_prefix}_cleaned.tsv", sep='\t', index=False)
    binary_df.to_csv(f"{output_prefix}_binary.tsv", sep='\t', index=False)
    
    for threshold, thresh_df in threshold_dfs.items():
        thresh_df.to_csv(f"{output_prefix}_{threshold}hot.tsv", sep='\t', index=False)
    
    # Calculate statistics
    above_counts = {threshold: len(df[df['recomb_rate'] > threshold]) for threshold in thresholds}
    
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
    
    for threshold in thresholds:
        above_count = above_counts[threshold]
        print(f"Rows with recombination rate > {threshold}: {above_count} ({above_count/len(df)*100:.2f}%)")
    
    print("\nBinary version (mean threshold):")
    print(f"Rows with recombination rate = 0: {len(binary_df[binary_df['recomb_rate'] == 0])}")
    print(f"Rows with recombination rate = 1: {len(binary_df[binary_df['recomb_rate'] == 1])}")
    
    for threshold in thresholds:
        thresh_df = threshold_dfs[threshold]
        print(f"\n{threshold}-hot version:")
        print(f"Rows with recombination rate = 0: {len(thresh_df[thresh_df['recomb_rate'] == 0])}")
        print(f"Rows with recombination rate = 1: {len(thresh_df[thresh_df['recomb_rate'] == 1])}")
    
    print("\nOutput saved to:")
    print(f"- {output_prefix}_cleaned.tsv (original recombination rates)")
    print(f"- {output_prefix}_binary.tsv (binary recombination rates)")
    
    for threshold in thresholds:
        print(f"- {output_prefix}_{threshold}hot.tsv ({threshold}-hot recombination rates)")
    
    return df

def main(genome_path, hotspot_file, rates_file, output_prefix):
    """Main function to run the entire pipeline"""
    print("Starting recombination hotspot processing pipeline...")
    
    # Load genome
    print(f"Loading genome from {genome_path}...")
    try:
        genome = Fasta(genome_path, as_raw=True)
        print("Loaded genome using pyfaidx")
    except Exception as e:
        print(f"Error loading genome: {str(e)}")
        sys.exit(1)
    
    # Step 1: Extract sequences and compute recombination rates
    combined_tsv = f"{output_prefix}_combined.tsv"
    extract_and_rate(genome, hotspot_file, rates_file, combined_tsv)
    
    # Step 2: Process hotspot data
    processed_tsv = f"{output_prefix}_processed.tsv"
    process_hotspots(combined_tsv, processed_tsv)
    
    # Step 3: Generate negative samples
    with_negatives_tsv = f"{output_prefix}_with_negatives.tsv"
    generate_negative_samples(genome, processed_tsv, with_negatives_tsv)
    
    # Step 4: Create thresholded versions
    create_thresholded_versions(with_negatives_tsv, output_prefix)
    
    print("\nEntire pipeline completed successfully!")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <genome_file> <hotspot_file> <rates_file> <output_prefix>")
        print("  genome_file: Path to the genome file (rn6.2bit or rn6.fa)")
        print("  hotspot_file: Path to hotspot table (GSE163474_allMergedRatHotspots.finalTable.tab)")
        print("  rates_file: Path to recombination rates file (FileS2.tab)")
        print("  output_prefix: Prefix for output files")
        sys.exit(1)
    
    main(*sys.argv[1:])