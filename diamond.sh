#!/bin/bash
# Script name: assign_OGs_precise.sh
# Usage: bash diamond.sh <input_directory> <diamond_db> <og_list> <output_csv> <matches_dir>
# Input: Directory with sample faa files and necessary paths
# Output: Presence/absence matrix and detailed alignment results

# Check if all required parameters are provided
if [ $# -ne 5 ]; then
    echo "Error: Missing required parameters"
    echo "Usage: bash $0 <input_directory> <diamond_db> <og_list> <output_csv> <matches_dir>"
    exit 1
fi

# Get parameters
INPUT_DIR="$1"
DIAMOND_DB="$2"
OG_LIST="$3"
OUTPUT_CSV="$4"
MATCHES_DIR="$5"
THREADS=10  # Default threads, can be adjusted

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist"
    exit 1
fi

# Check if essential files exist
if [ ! -f "$DIAMOND_DB" ]; then
    echo "Error: Diamond database file not found at '$DIAMOND_DB'"
    exit 1
fi

if [ ! -f "$OG_LIST" ]; then
    echo "Error: OG list file not found at '$OG_LIST'"
    exit 1
fi

# Create output directories
mkdir -p "$MATCHES_DIR"
mkdir -p $(dirname "$OUTPUT_CSV")

# Generate header
echo "Sample,$(tr '\n' ',' < "$OG_LIST" | sed 's/,$//')" > "$OUTPUT_CSV"

# Check if any FAA files exist in the input directory
FAA_COUNT=$(ls -1 "$INPUT_DIR"/*.faa 2>/dev/null | wc -l)
if [ "$FAA_COUNT" -eq 0 ]; then
    echo "Error: No FAA files found in '$INPUT_DIR'"
    exit 1
fi

echo "Found $FAA_COUNT FAA files in '$INPUT_DIR'"
echo "Starting analysis..."

# Process each sample
for faa in "$INPUT_DIR"/*.faa; do
    sample=$(basename "$faa" .faa)
    echo "Processing sample: $sample"
    
    # DIAMOND alignment
    diamond blastp \
        -d "$DIAMOND_DB" \
        -q "$faa" \
        -o "$MATCHES_DIR/${sample}.tsv" \
        --evalue 1e-5 \
        --id 60 \
        --query-cover 75 \
        --subject-cover 60 \
        --max-target-seqs 5 \
        --threads "$THREADS" \
        --outfmt 6 qseqid sseqid pident qcovhsp
    
    # Post-filtering and OG assignment
    awk -v sample="$sample" '
    BEGIN {
        FS = "\t";
        OFS = ",";
        # Load OG list
        while ((getline og < ARGV[1]) > 0) {
            og_in_list[og] = 1;
        }
        close(ARGV[1]);
    }
    {
        # Extract query gene and OG name
        query = $1;
        split($2, arr, "|");
        og = arr[1];
        # Focus only on OGs in the target list
        if (og in og_in_list) {
            # Record best match (sorted by bitscore)
            if (!(query in best_match) || $12 > best_score[query]) {
                best_match[query] = og;
                best_score[query] = $12;
            }
        }
    }
    END {
        # Count covered genes for each OG
        for (query in best_match) {
            og = best_match[query];
            coverage[og]++;
        }
        # Generate presence/absence markers
        line = sample;
        while ((getline og < ARGV[1]) > 0) {
            presence = (og in coverage && coverage[og] >= 1) ? 1 : 0;
            line = line OFS presence;
        }
        print line;
    }' "$OG_LIST" "$MATCHES_DIR/${sample}.tsv" >> "$OUTPUT_CSV"
done

echo "Analysis complete!"
echo "Presence/absence matrix saved to: $OUTPUT_CSV"