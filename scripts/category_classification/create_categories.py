import json
import pandas as pd
import os
import argparse

def prepare_category_csvs(input_file, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the JSON file line by line
    print(f"Reading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = {}
        for line in f.readlines():
            record = json.loads(line)
            for k, v in record.items():
                if k not in data:
                    data[k] = []
                data[k].append(v)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save the complete dataset
    complete_data_path = os.path.join(output_dir, 'complete_data.csv')
    df.to_csv(complete_data_path, index=False)
    print(f"Complete dataset saved to {complete_data_path}")
    
    # Get unique categories
    categories = df.category.unique()
    print(f"Found {len(categories)} unique categories")
    
    # Create a mapping for category names (replace spaces with underscores)
    categories_col = {category: category.replace(' ', '_') for category in categories}
    
    # Drop unnecessary columns
    df2 = df.drop(['link', 'authors', 'date'], axis=1) if all(col in df.columns for col in ['link', 'authors', 'date']) else df
    
    # One-hot encode the categories
    one_hot = pd.get_dummies(df2['category'])
    one_hot = one_hot.rename(columns=categories_col)
    
    # Combine with the original data
    df3 = pd.concat([df2.drop('category', axis=1), one_hot], axis=1)
    
    # Create separate CSV files for each category
    for category, category_col in categories_col.items():
        print(f"Processing category: {category}")
        # Select relevant columns
        tmp = df3[['headline', 'short_description', category_col]]
        # Rename the category column
        tmp = tmp.rename(columns={category_col: 'category'})
        # Save to CSV
        output_path = os.path.join(output_dir, f"{category_col}.csv")
        tmp.to_csv(output_path, index=False)
        print(f"Saved {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare CSV files for different news categories')
    parser.add_argument('input_file', default='data/category_classification/News_Category_Dataset_v3.json', type=str, help='Path to the input JSON file')
    parser.add_argument('output_dir', default='data/category_classification/category_csv', type=str, help='Directory to save the output CSV files')
    
    args = parser.parse_args()
    
    prepare_category_csvs(args.input_file, args.output_dir)
    print("All category CSV files have been generated successfully!")
