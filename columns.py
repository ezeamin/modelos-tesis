import pandas as pd

def main():
    # Change the filename to the path of your CSV file
    filename = 'datasets/CEAS_08.csv'

    # Read CSV file using pandas
    try:
        df = pd.read_csv(filename)
        # Find and print a row where label equals 0
        row_with_label_0 = df[df['label'] == 0].head(1)
        if not row_with_label_0.empty:
            print("Row where label=0:")
            print(row_with_label_0.to_string(index=False))
        else:
            print("No row found where label=0.")
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except pd.errors.EmptyDataError:
        print(f"File '{filename}' is empty.")
    except pd.errors.ParserError:
        print(f"Unable to parse file '{filename}'. It might not be a valid CSV file.")

if __name__ == "__main__":
    main()
