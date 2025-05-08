#!/usr/bin/env python3
import pandas as pd
import numpy as np

def main():
    # Read the dataset
    print("Reading dataset...")
    df = pd.read_csv('processed_hg19_with_negatives.tsv', sep='\t')
    
    # Count initial rows
    initial_rows = len(df)
    
    # Fill empty strength values with 0
    print("Filling empty strength values...")
    df['strength'] = df['strength'].fillna(0)
    
    # Remove rows with no recombination rate
    print("Removing rows with no recombination rate...")
    df = df.dropna(subset=['recomb_rate'])
    
    # Shift strength values for non-zero entries (starting at 1)
    print("Shifting strength values...")
    # Get the minimum strength value from non-zero entries
    min_strength = df[df['strength'] != 0]['strength'].min()
    
    # For non-zero entries, shift up to make 1 the minimum
    df.loc[df['strength'] != 0, 'strength'] = df[df['strength'] != 0]['strength'] - min_strength
    
    # Create binary recombination rate version
    print("Creating binary recombination rate version...")
    mean_recomb = df['recomb_rate'].mean()
    binary_df = df.copy()
    binary_df['recomb_rate'] = (binary_df['recomb_rate'] > mean_recomb).astype(int)
    
    # Create thresholded versions
    print("Creating thresholded versions...")
    three_hot_df = df.copy()
    three_hot_df['recomb_rate'] = (three_hot_df['recomb_rate'] > 3).astype(int)
    
    four_hot_df = df.copy()
    four_hot_df['recomb_rate'] = (four_hot_df['recomb_rate'] > 4).astype(int)
    
    five_hot_df = df.copy()
    five_hot_df['recomb_rate'] = (five_hot_df['recomb_rate'] > 5).astype(int)
    
    # Calculate statistics
    above_3 = len(df[df['recomb_rate'] > 3])
    above_4 = len(df[df['recomb_rate'] > 4])
    above_5 = len(df[df['recomb_rate'] > 5])
    
    # Save all versions
    print("Saving datasets...")
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
    print(f"Rows with recombination rate > 3: {above_3} ({above_3/len(df)*100:.2f}%)")
    print(f"Rows with recombination rate > 4: {above_4} ({above_4/len(df)*100:.2f}%)")
    print(f"Rows with recombination rate > 5: {above_5} ({above_5/len(df)*100:.2f}%)")
    
    print("\nBinary version (mean threshold):")
    print(f"Rows with recombination rate = 0: {len(binary_df[binary_df['recomb_rate'] == 0])}")
    print(f"Rows with recombination rate = 1: {len(binary_df[binary_df['recomb_rate'] == 1])}")
    
    print("\n3-hot version:")
    print(f"Rows with recombination rate = 0: {len(three_hot_df[three_hot_df['recomb_rate'] == 0])}")
    print(f"Rows with recombination rate = 1: {len(three_hot_df[three_hot_df['recomb_rate'] == 1])}")
    
    print("\n4-hot version:")
    print(f"Rows with recombination rate = 0: {len(four_hot_df[four_hot_df['recomb_rate'] == 0])}")
    print(f"Rows with recombination rate = 1: {len(four_hot_df[four_hot_df['recomb_rate'] == 1])}")
    
    print("\n5-hot version:")
    print(f"Rows with recombination rate = 0: {len(five_hot_df[five_hot_df['recomb_rate'] == 0])}")
    print(f"Rows with recombination rate = 1: {len(five_hot_df[five_hot_df['recomb_rate'] == 1])}")
    
    print("\nOutput saved to:")
    print("- processed_hg19_with_negatives_cleaned.tsv (original recombination rates)")
    print("- processed_hg19_with_negatives_binary.tsv (binary recombination rates)")
    print("- processed_hg19_with_negatives_3hot.tsv (3-hot recombination rates)")
    print("- processed_hg19_with_negatives_4hot.tsv (4-hot recombination rates)")
    print("- processed_hg19_with_negatives_5hot.tsv (5-hot recombination rates)")

if __name__ == "__main__":
    main() 