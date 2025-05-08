import pandas as pd

# Read the CSV file
df = pd.read_csv('final_mouse.csv')

# Select and rename the columns
df_processed = df[['cs', 'start', 'end', 'dna_seq', 'prdm_norm', '4Ner/kb']].copy()
df_processed.columns = ['chrom', 'start', 'end', 'sequence', 'strength', 'recomb_rate']

# Multiply recomb_rate by 100 to convert 4Ner/kb to cM/Mb
df_processed.loc[:, 'recomb_rate'] = df_processed['recomb_rate'] * 100

# Save as TSV
df_processed.to_csv('processed_mouse.tsv', sep='\t', index=False)

print("Data processing complete. Output saved as 'processed_mouse.tsv'")