import wrds
import pandas as pd

db = wrds.Connection()

q = """
SELECT COUNT(*) AS n_trades
FROM taqm_2023.ctm_20230111
WHERE sym_root = 'PLD'
"""
print(db.raw_sql(q))
