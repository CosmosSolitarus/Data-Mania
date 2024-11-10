import csv

with open("us_accidents_sample_cleaned.csv", "r") as file:
    reader = csv.reader(file)
    
    # Get columns (first row in the file)
    columns = next(reader)
    
    # Get first data row (second row in the file)
    first_data_row = next(reader)
    
    # Count remaining rows, adding 1 to include the first data row
    row_count = 1 + sum(1 for _ in file)

# Create a dictionary for the first data row
first_data_dict = dict(zip(columns, first_data_row))

# Print results
print("Columns:", columns)
print("Number of rows:", row_count)
print("Number of columns:", len(columns))
print("First data row (as dictionary):")
for column, value in first_data_dict.items():
    print(f'"{column}": [{value}],')
