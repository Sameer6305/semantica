import os
import tempfile

from semantica.ingest.pandas_ingestor import PandasIngestor


# -------------------------------------------------------
# Helper
# -------------------------------------------------------

def write_temp_csv(content: str, encoding="utf-8"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.close()
    with open(tmp.name, "w", encoding=encoding) as f:
        f.write(content)
    return tmp.name


# =======================================================
# ENCODING (5 tests)
# =======================================================

def test_encoding_latin1():
    content = "name,city\nJosé,São Paulo\n"
    path = write_temp_csv(content, encoding="latin-1")
    data = PandasIngestor().from_csv(path)
    assert data.dataframe.iloc[0]["name"] == "José"
    os.remove(path)


def test_encoding_utf8():
    content = "user,country\n李雷,China\n"
    path = write_temp_csv(content, encoding="utf-8")
    data = PandasIngestor().from_csv(path)
    assert data.dataframe.iloc[0]["user"] == "李雷"
    os.remove(path)


def test_encoding_accented_text():
    content = "company,city\nRenée,Zürich\n"
    path = write_temp_csv(content, encoding="latin-1")
    data = PandasIngestor().from_csv(path)
    assert data.row_count == 1
    os.remove(path)


def test_encoding_spanish():
    content = "org,country\nTelefónica,España\n"
    path = write_temp_csv(content, encoding="latin-1")
    data = PandasIngestor().from_csv(path)
    assert data.row_count == 1
    os.remove(path)


def test_encoding_ansi_cp1252():
    content = "brand,city\nPeugeot,Montréal\n"
    path = write_temp_csv(content, encoding="cp1252")   
    data = PandasIngestor().from_csv(path)
    assert data.dataframe.iloc[0]["city"] == "Montréal"
    os.remove(path)


# =======================================================
# DELIMITERS (4 tests)
# =======================================================

def test_delimiter_comma():
    content = "a,b\n1,2\n"
    path = write_temp_csv(content)
    data = PandasIngestor().from_csv(path)
    assert list(data.columns) == ["a", "b"]
    os.remove(path)


def test_delimiter_semicolon():
    content = "a;b\n1;2\n"
    path = write_temp_csv(content)
    data = PandasIngestor().from_csv(path)
    assert list(data.columns) == ["a", "b"]
    os.remove(path)


def test_delimiter_pipe():
    content = "a|b\n1|2\n"
    path = write_temp_csv(content)
    data = PandasIngestor().from_csv(path)
    assert list(data.columns) == ["a", "b"]
    os.remove(path)


def test_delimiter_tab():
    content = "a\tb\n1\t2\n"
    path = write_temp_csv(content)
    data = PandasIngestor().from_csv(path)
    assert list(data.columns) == ["a", "b"]
    os.remove(path)


# =======================================================
# BAD ROWS (3 tests)
# =======================================================

def test_bad_row_extra_columns():
    content = "x,y\n1,2\n1,2,3,4\n5,6\n"
    path = write_temp_csv(content)
    data = PandasIngestor().from_csv(path)
    assert data.row_count == 2
    os.remove(path)

def test_bad_row_missing_column():
    content = "x,y\n1,2\n3\n4,5\n"
    path = write_temp_csv(content)
    data = PandasIngestor().from_csv(path)
    assert data.row_count == 3
    assert data.dataframe["y"].isna().sum() == 1

def test_bad_row_unclosed_quote():
    content = "x,y\n1,2\n\"3,4\n5,6\n"
    path = write_temp_csv(content)
    data = PandasIngestor().from_csv(path)
    assert data.row_count == 3



