-- runs lineage
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  dag_id TEXT,
  ts_started TIMESTAMPTZ DEFAULT NOW(),
  ts_finished TIMESTAMPTZ,
  status TEXT,
  model_version TEXT,
  data_snapshot_uri TEXT,
  preds_uri TEXT
);

-- results table (serving)
CREATE TABLE IF NOT EXISTS classifications (
  id BIGSERIAL PRIMARY KEY,
  doc_id TEXT NOT NULL,
  source_file TEXT,
  predicted_type TEXT NOT NULL,
  dated TEXT,
  needs_human_review BOOLEAN NOT NULL DEFAULT FALSE,
  key_facts JSONB DEFAULT '[]'::jsonb,
  confidence_reasoning TEXT,
  model_version TEXT NOT NULL,
  run_id TEXT REFERENCES runs(run_id),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (doc_id, model_version)
);

CREATE INDEX IF NOT EXISTS idx_classifications_label ON classifications(predicted_type);
CREATE INDEX IF NOT EXISTS idx_classifications_created ON classifications(created_at);
CREATE INDEX IF NOT EXISTS idx_classifications_doc ON classifications(doc_id);

