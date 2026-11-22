-- Shorts Intelligence: public YouTube Shorts researched per game, with every metric
-- collection kept as a snapshot so changes can be tracked across refreshes.
CREATE TABLE research_videos (
    video_id      TEXT PRIMARY KEY,
    game          TEXT NOT NULL,
    data          TEXT NOT NULL,  -- title, channel, published_at, duration, tags, queries...
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL
);
CREATE INDEX research_videos_game ON research_videos(game);

CREATE TABLE research_snapshots (
    video_id      TEXT NOT NULL REFERENCES research_videos(video_id) ON DELETE CASCADE,
    collected_at  TEXT NOT NULL,
    views         INTEGER,
    likes         INTEGER,  -- NULL when the creator hides likes
    comments      INTEGER,  -- NULL when comments are off
    channel_subs  INTEGER,  -- NULL when the channel hides its count
    PRIMARY KEY (video_id, collected_at)
);

CREATE TABLE research_runs (
    id            TEXT PRIMARY KEY,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    status        TEXT NOT NULL,  -- running, done, failed
    data          TEXT NOT NULL   -- games, queries, found, new, quota used, error
);
