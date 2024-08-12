import diskcache
import docx
import dotenv
import openai
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cosine

dotenv.load_dotenv()

DISKCACHE = diskcache.FanoutCache(directory=".diskcache", timeout=1, size_limit=1024**3)

INPUT_FILE = "assets/1-s2.0-S221345302400096X-mmc1.docx"
OUTPUT_FILE_MAIN = "docs/data/table1.json"
OUTPUT_FILE_COOKING_METHOD = "docs/data/table1_by_cooking_method.json"

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

@DISKCACHE.memoize(tag="get_embedding")
def get_embedding(text: str) -> list[float]:
    """Return the embedding vector for the given text."""
    print(f"Getting embedding for {text!r}.")
    client = openai.OpenAI()
    response = client.embeddings.create(input=text, model="text-embedding-3-large")
    embedding = response.data[0].embedding
    return embedding

def get_food_centralities(texts: list[str], *, scale: bool = False, sort: bool = False) -> dict[str, float]:
    """Return the food centralities of the specified texts in the range of 0 to 100.

    Higher values indicate greater centrality to food.

    The centralities are computed based on one minus the cosine distance of the text embeddings from the reference embedding of "Food".
    
    Args:
        texts: List of texts to compute centralities for.
        scale: Whether to minmax scale the distances to use the entire output range. Default is False.
        sort: Whether to sort the centralities in descending order. Default is False.
    """
    reference_embedding = get_embedding("Food")
    text_embeddings = {text: get_embedding(text) for text in texts}
    distances = {text: float(cosine(reference_embedding, text_embedding)) for text, text_embedding in text_embeddings.items()}  # As a dict for debugging purposes.
    distances = [d for d in distances.values()]  # As a list for optional scaling.
    print(f"Unscaled embedding distance range of foods is from {min(distances):.2f} to {max(distances):.2f}.")
    assert all(0 <= d <= 1 for d in distances)  # Required for computing centrality as `1 - distance`. If this fails, either mandate scaling, or compute centrality as `1 / (1 + distance)`.

    if scale:
        distances = MinMaxScaler(clip=True).fit_transform(np.array(list(distances)).reshape(-1, 1)).flatten().tolist()  # `clip=True`` is used to avoid a value such as 1.0000000000000002.
        assert all(0 <= d <= 1 for d in distances), distances
        print(f"Scaled embedding distance range of foods is from {min(distances):.2f} to {max(distances):.2f}.")
    distances = {text: distance for text, distance in zip(texts, distances)}
    centralities = {text: 1 - distance for text, distance in distances.items()}
    assert all(0 <= c <= 1 for c in centralities.values())
    centralities = {text: centrality * 100 for text, centrality in centralities.items()}  # Convert to percentage.
    assert all(0 <= c <= 100 for c in centralities.values())
    print(f"Embedding centralities range of foods is from {min(centralities.values()):.2f} to {max(centralities.values()):.2f}.")
    if sort:
        centralities = dict(sorted(centralities.items(), key=lambda item: item[1], reverse=True))
    return centralities


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
    df.sort_values(["Food", "Specification"], ignore_index=True, inplace=True)
    
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
    
    df.rename(columns={"Food": "Category", "Specification": "Food"}, inplace=True)
    
    # food_centralities = get_food_centralities(df["Food"].tolist(), scale=True, sort=True)
    # df["FoodCentrality"] = df["Food"].map(food_centralities)
    # df.sort_values("FoodCentrality", ascending=False, inplace=True)
    
    df = df[[c for c in df.columns if c not in AGE_NAMES] + AGE_NAMES]  # Move individual AGE columns to the end.
    print(f"Converted table data to dataframe with shape {"x".join(map(str, df.shape))}.")
    return df

def group_table1_by_cooking_method(df: pd.DataFrame) -> pd.DataFrame:
    """Return the table dataframe grouped by cooking method."""
    print(f"Grouping table dataframe by cooking method.")
    df = df[['Food', 'Total', 'CrosslinkingSubtotal', 'NonCrosslinkingSubtotal']].copy()

    cooking_methods = ['boiled', 'caramelized', 'deep-fried', 'pan-fried', 'pasteurized', 'pickled', 'raw', 'roasted', 'steamed', 'stewed', 'stir-fried', 'UHT']

    def extract_cooking_method(food: str) -> str | pd.NA.__class__:
        for method in cooking_methods:
            if food.endswith((f' ({method})', ' {method})')):
                return method
        return pd.NA

    df['CookingMethod'] = df['Food'].apply(extract_cooking_method)
    df.dropna(subset='CookingMethod', ignore_index=True, inplace=True)
    df_grouped = df.groupby("CookingMethod").agg(
        Count=pd.NamedAgg(column='Food', aggfunc='count'),
        AvgTotal=pd.NamedAgg(column='Total', aggfunc='mean'),
        AvgCrosslinkingSubtotal=pd.NamedAgg(column='CrosslinkingSubtotal', aggfunc='mean'),
        AvgNonCrosslinkingSubtotal=pd.NamedAgg(column='NonCrosslinkingSubtotal', aggfunc='mean')
    ).reset_index()
    df_grouped.sort_values("CookingMethod", ignore_index=True, key=lambda series: series.str.lower(), inplace=True)
    print(f"Grouped table dataframe by cooking method with shape {"x".join(map(str, df_grouped.shape))}.")
    return df_grouped


def write_dataframe_to_json_file(df: pd.DataFrame, output_file: str) -> None:
    """Write the dataframe to the specified JSON file.

    Args:
        df: Dataframe
        output_file: Path to the output JSON file.
    """
    print(f"Writing dataframe with {len(df):,} rows to JSON file {output_file}")
    df.to_json(output_file, orient="records", double_precision=2, force_ascii=False, indent=2)
    print(f"Wrote dataframe to JSON file {output_file}.")


def main() -> None:
    data = read_table_from_docx_file(INPUT_FILE, table_index=0)

    df = convert_table1_data_to_dataframe(data)
    write_dataframe_to_json_file(df, OUTPUT_FILE_MAIN)

    df_grouped = group_table1_by_cooking_method(df)
    # print(df_grouped)
    write_dataframe_to_json_file(df_grouped, OUTPUT_FILE_COOKING_METHOD)


if __name__ == "__main__":
    main()
