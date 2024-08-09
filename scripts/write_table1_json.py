import docx
import pandas as pd

INPUT_FILE = "assets/1-s2.0-S221345302400096X-mmc1.docx"
OUTPUT_FILE = "web/table1.json"

CROSSLINKING_STATUS: dict[str, bool] = {  # From "Supplementary Figure 1" in input file. Order is as per "Supplementary Table 1" however.
    "CML": False,
    "CEL": False,
    "GOLD": True,
    "G-H1": False,
    "MOLD": True,
    "MG-H2": False,  # Not documented in the figure, but presumed to be the same as for "MG-H1".
    "MG-H1/3": False,  # Documented only for "MG-H1" in the figure, not for "MG-H3", but presumed to be the same.
    "Pentosidine": True,
    "Argpyrimidine": False,
}
AGE_NAMES: list[str] = list(CROSSLINKING_STATUS)


def set_pandas_display_options() -> None:
    """Set pandas display options."""
    # Ref: https://stackoverflow.com/a/52432757/
    display = pd.options.display

    display.max_columns = 1000
    display.max_rows = 10_000
    display.max_colwidth = 199
    display.width = 1000
    display.precision = 2
    display.float_format = lambda x: '{:,.2f}'.format(x)  # Ref: https://stackoverflow.com/a/47614756/


set_pandas_display_options()


def read_table_from_docx_file(file_path: str, *, table_index: int = 0) -> list[list[str]]:
    """Return the table of a docx file as a list of lists.

    The outer list represents the rows of the table, and the inner lists represent the cells in each row.

    Args:
        file_path: Path to the doc file.
        table_index: Index of the table in the document to be read. Its default is 0.
    """
    print(f"Loading table {table_index} from docx file {file_path}.")
    doc = docx.Document(file_path)
    table = doc.tables[table_index]
    data = [[cell.text.strip() for cell in row.cells] for row in table.rows]
    print(f"Loaded table {table_index} with {len(data):,} rows.")
    return data

def convert_table1_data_to_dataframe(data: list[list[str]]) -> pd.DataFrame:
    """Return the table data as a dataframe.

    Args:
        data: Table data to convert.
    """
    print(f"Converting table data with {len(data):,} rows to dataframe.")
    columns = data[0]
    df = pd.DataFrame(data[1:], columns=columns)
    df.sort_values(["Food", "Specification"], inplace=True)
    
    df[AGE_NAMES] = df[AGE_NAMES].astype(float)

    col_groups: dict[str, list[str]] = {
        "Total": AGE_NAMES,
        "Crosslinking": [col for col in AGE_NAMES if CROSSLINKING_STATUS[col]],
        "NonCrosslinking": [col for col in AGE_NAMES if not CROSSLINKING_STATUS[col]],
    }
    for group, cols in col_groups.items():
        total_col = "Total" if group == "Total" else f"{group}Subtotal"
        df[total_col] = df[cols].sum(axis=1)
        df[f"{group}Percentile"] = df[total_col].rank(pct=True) * 100
        df[f"{group}Zscore"] = (df[total_col] - df[total_col].mean()) / df[total_col].std()
    
    df = df[[c for c in df.columns if c not in AGE_NAMES] + AGE_NAMES]  # Move individual AGE columns to the end.
    df.rename(columns={"Food": "Category", "Specification": "Food"}, inplace=True)

    print(f"Converted table data to dataframe with shape {"x".join(map(str, df.shape))}.")
    return df


def write_table1_dataframe_to_json_file(df: pd.DataFrame, output_file: str) -> None:
    """Write the table dataframe to the specified JSON file.

    Args:
        df: Table dataframe
        output_file: Path to the output JSON file.
    """
    print(f"Writing table dataframe with {len(df):,} rows to JSON file {output_file}")
    df.to_json(output_file, orient="records", double_precision=2, force_ascii=False, indent=2)
    print(f"Wrote table dataframe to JSON file.")


def main() -> None:
    data = read_table_from_docx_file(INPUT_FILE, table_index=0)
    df = convert_table1_data_to_dataframe(data)
    write_table1_dataframe_to_json_file(df, OUTPUT_FILE)


if __name__ == "__main__":
    main()
