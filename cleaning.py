import pandas as pd
import numpy as np

"""
Safe Cleaning Pipeline:
1. Standardize column names
2. Remove empty rows and columns
3. Remove duplicates
4. Strip and lowercase text columns
5. Clean specific columns (gender, education, policy type, state names, etc.)
6. Clean and convert numeric columns
7. Handle missing values
"""

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names: lowercase with underscores."""
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r'[^a-z0-9]+', '_', regex=True)
        .str.replace(r'_+', '_', regex=True)
        .str.strip('_')
    )
    return df

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename specific columns for consistency."""
    df = df.rename(columns={'st': 'state'})
    return df

def remove_empty_rows_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove fully empty rows and columns."""
    return df.dropna(axis=0, how='all').dropna(axis=1, how='all')

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    return df.drop_duplicates()

def strip_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and convert text columns to lowercase."""
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].astype(str).str.strip().str.lower()
    return df

def clean_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize gender values to 'm' or 'f'."""
    if 'gender' in df.columns:
        df['gender'] = (
            df['gender']
            .replace({
                'male': 'm',
                'm': 'm',
                'female': 'f',
                'f': 'f',
                'femal': 'f'
            })
        )
    return df

def clean_education(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize education values."""
    if 'education' in df.columns:
        mapping = {
            'master': 'master',
            'bachelor': 'bachelor',
            'bachelors': 'bachelor',
            'high school or below': 'high school',
            'college': 'college',
            'doctor': 'doctorate'
        }
        df['education'] = df['education'].replace(mapping)
    return df

def clean_state_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize state names."""
    if 'state' in df.columns:
        df['state'] = df['state'].replace({
            'az': 'arizona',
            'wa': 'washington',
            'cali': 'california'
        })
    return df

def cap_monthly_premium_auto(df: pd.DataFrame, cap: float = 1000) -> pd.DataFrame:
    """Cap monthly_premium_auto at a maximum value."""
    if 'monthly_premium_auto' in df.columns:
        df['monthly_premium_auto'] = np.where(
            df['monthly_premium_auto'] > cap,
            cap,
            df['monthly_premium_auto']
        )
    return df

def clean_policy_type(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize policy_type values."""
    if 'policy_type' in df.columns:
        df['policy_type'] = df['policy_type'].replace({
            'personal auto': 'personal',
            'corporate auto': 'corporate',
            'special auto': 'special'
        })
    return df

def clean_customer_lifetime_value(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and convert customer_lifetime_value column."""
    if 'customer_lifetime_value' in df.columns:
        df['customer_lifetime_value'] = (
            df['customer_lifetime_value']
            .astype(str)
            .str.replace('%', '', regex=False)
            .str.strip()
        )
        df['customer_lifetime_value'] = pd.to_numeric(df['customer_lifetime_value'], errors='coerce')
    return df

def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to numeric or datetime where possible."""
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col], errors='ignore')
            except:
                pass
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values: numeric=0, text='unknown'."""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(0, inplace=True)
        else:
            df[col].fillna('unknown', inplace=True)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline."""
    df = clean_column_names(df)
    df = rename_columns(df)
    df = remove_empty_rows_columns(df)
    df = remove_duplicates(df)
    df = strip_text_columns(df)            # strip + lowercase text before mapping
    df = clean_gender(df)
    df = clean_education(df)
    df = clean_state_names(df)
    df = cap_monthly_premium_auto(df)
    df = clean_policy_type(df)
    df = clean_customer_lifetime_value(df)
    df = convert_data_types(df)
    df = handle_missing_values(df)
    return df
