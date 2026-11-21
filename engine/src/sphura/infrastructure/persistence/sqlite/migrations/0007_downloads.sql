-- YouTube download queue and library. Kept apart from projects: a download becomes a
-- project only when the user clicks Edit Video.
CREATE TABLE downloads (
    id             TEXT PRIMARY KEY,
    video_id       TEXT NOT NULL,
    url            TEXT NOT NULL,
    title          TEXT NOT NULL,
    channel        TEXT NOT NULL DEFAULT '',
    duration_s     REAL NOT NULL DEFAULT 0,
    thumbnail_url  TEXT NOT NULL DEFAULT '',
    thumbnail_path TEXT,
    playlist_title TEXT,
    quality        TEXT NOT NULL,
    status         TEXT NOT NULL,  -- queued, downloading, paused, completed, failed, cancelled
    progress       REAL NOT NULL DEFAULT 0,
    error          TEXT,
    file_path      TEXT,
    size_bytes     INTEGER,
    width          INTEGER,
    height         INTEGER,
    fps            REAL,
    project_id     TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    completed_at   TEXT
);
CREATE INDEX ix_downloads_video ON downloads(video_id);
CREATE INDEX ix_downloads_status ON downloads(status);
