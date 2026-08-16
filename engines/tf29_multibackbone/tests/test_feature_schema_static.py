"""Contract and regression checks for test feature schema static in the TF2.9 multi-backbone engine."""

import json
from pathlib import Path
import pandas as pd

def test_schema():
 p=Path(__file__).resolve().parents[1]/"materon/features"; raw=pd.read_csv(p/"physic_features_pool.csv"); ctx=pd.read_csv(p/"grouped_pca_context.csv"); meta=json.loads((p/"feature_metadata.json").read_text())
 assert raw.shape==(100,21); assert ctx.shape==(100,13); assert len(meta["raw_feature_columns"])==20; assert len(meta["groups"])==12
