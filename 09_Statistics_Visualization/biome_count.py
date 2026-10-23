import pandas as pd

def generate_biome_statistics(input_path, output_path):
    # Load the CSV file. Explicitly define NA strings based on the source data.
    df = pd.read_csv(input_path, na_values=['NA', 'NaN', ''])

    # Define the list of variables to count based on the input structure
    target_vars = [
        'DBH', 'H', 'AGE', 'BA', 'Trees_N', 
        'AGB', 'V', 'shannon', 'CV_DBH'
       ]

    # Map original column names to the expected output column names
    rename_mapping = {
        'DBH': 'DBH',
        'H': 'Height',
        'AGE': 'AGE',
        'BA': 'BA',
        'Trees_N': 'Density',
        'AGB': 'AGB',
        'V': 'GSV',
        'shannon': 'Shannon',
        'CV_DBH': 'CV DBH'
    }

    # Count total rows (samples) per BIOME
    total_samples = df.groupby('BIOME').size().reset_index(name='Total_Samples')

    # Count valid (non-null) entries for specific target variables per BIOME
    valid_samples = df.groupby('BIOME')[target_vars].count().reset_index()

    # Rename the columns to match the target table format
    valid_samples = valid_samples.rename(columns=rename_mapping)

    # Merge the total counts with the valid variable counts
    summary_table = pd.merge(total_samples, valid_samples, on='BIOME')

    # Sort by BIOME to ensure ascending order
    summary_table = summary_table.sort_values(by='BIOME').reset_index(drop=True)

    # Save the resulting summary table to a new CSV file
    summary_table.to_csv(output_path, index=False)
    
    return summary_table

if __name__ == "__main__":
    input_file = "/data/global_mapping/plotdata_CV/fina_plot_with_RS.csv"
    output_file = "/data/global_mapping/models_V100/Complete/biome_statistics_summary.csv"
    
    result_df = generate_biome_statistics(input_file, output_file)
    print(result_df.to_string())