CREATE TABLE projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    mode        TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE media_assets (
    id             TEXT PRIMARY KEY,
    project_id     TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_uri     TEXT NOT NULL,
    storage_kind   TEXT NOT NULL,
    container      TEXT NOT NULL,
    size_bytes     INTEGER NOT NULL,
    fingerprint    TEXT NOT NULL,
    original_name  TEXT NOT NULL,
    sha256         TEXT,
    media_info     TEXT,
    created_at     TEXT NOT NULL
);
CREATE INDEX idx_media_assets_project ON media_assets(project_id);
CREATE INDEX idx_media_assets_fingerprint ON media_assets(fingerprint);

CREATE TABLE jobs (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    type          TEXT NOT NULL,
    status        TEXT NOT NULL,
    stage         TEXT,
    progress      REAL NOT NULL DEFAULT 0,
    payload       TEXT NOT NULL,
    result        TEXT,
    error         TEXT,
    attempts      INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT,
    heartbeat_at  TEXT
);
CREATE INDEX idx_jobs_status_created ON jobs(status, created_at);
CREATE INDEX idx_jobs_project ON jobs(project_id);
