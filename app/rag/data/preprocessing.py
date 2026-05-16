import pandas as pd
import ast

def parse_age_to_months(raw):
    try:
        parsed = ast.literal_eval(raw)
        value, unit = parsed[0][0], parsed[0][1]
        if unit == "year":
            return value * 12
        elif unit == "month":
            return value
        elif unit == "day":
            return value / 30.0
        return None
    except (ValueError, SyntaxError, IndexError, TypeError):
        return None

df = pd.read_csv("app/rag/data/PMC-Patients.csv")
df2 = df[['patient_uid', 'title', 'patient', 'age', 'gender']].copy()
df2['age'] = df2['age'].apply(parse_age_to_months)
df2 = df2[:3904]
df2.to_csv("app/rag/data/preprocessed.csv", index=False)

