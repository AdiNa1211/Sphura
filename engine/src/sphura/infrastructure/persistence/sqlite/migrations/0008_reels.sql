-- AI highlight reels: many source videos -> one edited compilation. The reel's sources,
-- detected moments, selection, settings and output live in one JSON document.
CREATE TABLE reels (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    status      TEXT NOT NULL,  -- importing, analysing, review, rendering, done, failed
    data        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
