CREATE TABLE clips (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    asset_id      TEXT NOT NULL REFERENCES media_assets(id) ON DELETE CASCADE,
    candidate_id  TEXT,
    title         TEXT NOT NULL,
    edl           TEXT NOT NULL,
    version       INTEGER NOT NULL DEFAULT 1,
    output_path   TEXT,
    rendered_at   TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX idx_clips_project ON clips(project_id);
