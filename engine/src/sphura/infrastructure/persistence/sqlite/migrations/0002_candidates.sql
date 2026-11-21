CREATE TABLE candidates (
    id           TEXT PRIMARY KEY,
    project_id   TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    asset_id     TEXT NOT NULL REFERENCES media_assets(id) ON DELETE CASCADE,
    start_s      REAL NOT NULL,
    end_s        REAL NOT NULL,
    peak_s       REAL NOT NULL,
    signal_score REAL NOT NULL,
    peak_score   REAL NOT NULL,
    drivers      TEXT NOT NULL,
    features     TEXT NOT NULL,
    judgement    TEXT,
    created_at   TEXT NOT NULL
);
CREATE INDEX idx_candidates_project ON candidates(project_id);
CREATE INDEX idx_candidates_asset ON candidates(asset_id);
