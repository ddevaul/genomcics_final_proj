import pandas as pd




import pandas as pd
from Bio import SeqIO
import gzip

# REFERENCE GENOME concatenate files 
# --- Define chromosome names ---
chromosomes = [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY"]  # mouse only has 19 autosomes

# --- Load all chromosome sequences ---
chr_sequences = {}

for chrom in chromosomes:
    fasta_file = f"{chrom}.fa.gz"
    try:
        with gzip.open(fasta_file, "rt") as f:
            record = next(SeqIO.parse(f, "fasta"))
            chr_sequences[chrom] = str(record.seq)
        print(f"Loaded {chrom}")
    except FileNotFoundError:
        print(f"Warning: {fasta_file} not found. Skipping.")
        continue


# RECOMBINATION RATE DATA Fine-Scale Maps of Recombination Rates and Hotspots in the Mouse Genome

# Load your CSV file
df = pd.read_csv("supp_112.141036_TableS1 (1).csv")

# Convert Kb to base pairs
df['bp'] = (df['Kb'] * 1000).astype(int)

# Create start and end columns (per-SNP range)
df['start_bp'] = df['bp']
df['end_bp'] = df['bp'].shift(-1)  # next SNP's position

# Drop the temporary 'bp' column if you want
df.drop(columns=['bp'], inplace=True)

# Optional: remove the last row, which has no 'end_bp'
df = df.dropna(subset=['end_bp']) # should actually just make this to the last bp of the chromsome but idk what that is for mice






# MERGE HOTSPOT AND RECOMBINATION RATE DATA... TAKE AVERAGE RECOMBINATION RATE IF HOTSPOT SPANS MULTIPLE REGIONS
# Rename columns so we don’t shadow Python keywords
df_hotspot = df_hotspot.rename(columns={"from": "start", "to": "end"})
# (df already has start_bp, end_bp and 4Ner/kb)

# — Ensure numeric types —
df_hotspot[["start", "end"]] = df_hotspot[["start", "end"]].apply(pd.to_numeric, errors="coerce")
df[["start_bp", "end_bp", "4Ner/kb"]] = df[["start_bp", "end_bp", "4Ner/kb"]].apply(pd.to_numeric, errors="coerce")

# — Define a function that finds all overlaps and returns their mean rate —
def mean_rate_for_hotspot(row, df_subset):
    chrom = row["cs"]
    sub = df_subset[df_subset["Chr"] == chrom]
    mask = (sub["start_bp"] <= row["end"]) & (sub["end_bp"] >= row["start"])
    overl = sub.loc[mask, "4Ner/kb"]
    if overl.empty:
        return float("nan")
    return overl.mean()

# — Apply with progress logging every 5% —
total_rows = len(df_hotspot)
step = total_rows // 100  # 5% of total

results = []
for i, row in df_hotspot.iterrows():
    result = mean_rate_for_hotspot(row, df)
    results.append(result)
    if i % step == 0:
        percent = round((i / total_rows) * 100)
        print(f"Completed {percent}%")

df_hotspot["4Ner/kb"] = results

# — Optional: inspect or save —
print(df_hotspot.head())


df_hotspot.to_csv("hotspots_with_4Ner.csv", index=False)






# DI NUCLEOTIDE STUFF

import pandas as pd
from collections import Counter

# — 1) Standardize DNA sequence column name —
df_hotspot = df_hotspot.rename(columns={"DNA seq": "dna_seq"})

# — 2) Parameters for gapped k-mer —
k = 1
nucleotides = ["A", "C", "G", "T"]
dinucleotides = [a + b for a in nucleotides for b in nucleotides]

# — 3) Upper-case all sequences —
df_hotspot["dna_seq"] = df_hotspot["dna_seq"].astype(str).str.upper()

# — 4) Initialize frequency columns —
for dinuc in dinucleotides:
    df_hotspot[f"{dinuc}_k{k}"] = 0.0

# — 5) Function to compute gapped dinucleotide frequencies —
def compute_gapped_freqs(seq, k):
    counts = Counter()
    total = 0
    L = len(seq)
    for i in range(L - k - 1):
        a = seq[i]
        b = seq[i + k + 1]
        if a in nucleotides and b in nucleotides:
            counts[a + b] += 1
            total += 1
    return {
        dinuc: (counts[dinuc] / total if total > 0 else 0.0)
        for dinuc in dinucleotides
    }

# — 6) Apply with progress printing every 5% —
total_rows = len(df_hotspot)
step = max(1, total_rows // 20)  # 5% steps

for idx, row in df_hotspot.iterrows():
    freqs = compute_gapped_freqs(row["dna_seq"], k)
    for dinuc, freq in freqs.items():
        df_hotspot.at[idx, f"{dinuc}_k{k}"] = freq
    if idx % step == 0:
        percent = round((idx / total_rows) * 100)
        print(f"Completed {percent}%")

# — 7) (Optional) Save the updated DataFrame —
# df_hotspot.to_excel("hotspots_with_gapped_dinuc_freqs.xlsx", index=False)








import pandas as pd
import gzip
import re

# LOAD GENE ANNOTATION FILE


# --- Parse GTF to extract transcript-level entries with gene_name ---
records = []

with gzip.open("mm10.ensGene.gtf.gz", "rt") as gtf:
    for line in gtf:
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if f[2] == "transcript":
            chrom = f[0]
            s = int(f[3])
            e = int(f[4])
            m = re.search(r'gene_name "([^"]+)"', f[8])
            gn = m.group(1) if m else ""
            records.append((chrom, s, e, gn))

# --- Create DataFrame from transcripts ---
ann = pd.DataFrame(records, columns=["chromosome", "start", "end", "gene_name"])
ann['chromosome'] = ann['chromosome'].astype(str)








# --- Check for GENE overlaps IN HOTSPOT REGION and return 1/0 and gene names (joined by ;) ---
def check_overlap(row):
    chrom = row['cs']
    start = row['start']
    end = row['end']

    overlapping = ann[
        (ann['chromosome'] == chrom) &
        (ann['start'] <= end) &
        (ann['end'] >= start)
    ]

    if overlapping.empty:
        return pd.Series([0, ""])
    else:
        gene_names = ";".join(sorted(set(overlapping['gene_name'])))
        return pd.Series([1, gene_names])

# --- Apply to each row in hotspots with progress print every ~2% ---
total_rows = len(df_hotspot)
step = max(1, total_rows // 50)  # 2% of the total

# Collect results
overlap_results = []
for idx, row in df_hotspot.iterrows():
    result = check_overlap(row)
    overlap_results.append(result)
    if idx % step == 0:
        percent = round((idx / total_rows) * 100)
        print(f"Completed {percent}%")

# Assign results back
df_hotspot[['gene_overlap', 'gene_name']] = pd.DataFrame(overlap_results, index=df_hotspot.index)




# 1. Filter out chrX and chrY and make a copy
df_hotspot = df_hotspot[~df_hotspot['cs'].isin(['chrX', 'chrY'])].copy()

# 2. Take absolute value of 'ssdsPrdm9KO'
df_hotspot['ssdsPrdm9KO'] = df_hotspot['ssdsPrdm9KO'].abs()

# 3. Standardize the column into 'prdm_norm'
mean = df_hotspot['ssdsPrdm9KO'].mean()
std = df_hotspot['ssdsPrdm9KO'].std()
df_hotspot['prdm_norm'] = (df_hotspot['ssdsPrdm9KO'] - mean) / std
df_hotspot.to_csv("final_mouse.csv", index=False)



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