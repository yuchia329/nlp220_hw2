import polars as pl
from sklearn.model_selection import train_test_split

def processData(file_path):
    df = pl.read_csv(file_path)
    df.columns = ["category", "text"]
    category = df.get_column('category').unique(maintain_order=True)
    df = df.drop_nulls()
    df = df.with_columns(pl.col("category").cast(pl.Categorical))
    df = df.with_columns(pl.col("category").to_physical().alias("category"))
    categoryIndex = df.get_column('category').unique(maintain_order=True)
    x = df['text'].to_numpy()
    y = df['category'].to_numpy()
    return x, y, dict(map(lambda i,j : (i,j) , categoryIndex,category))

def splitData(X, y):
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    x_test, x_vali, y_test, y_vali = train_test_split(
        x_test, y_test, test_size=0.33, random_state=42)
    return x_train, x_vali, x_test, y_train, y_vali, y_test
