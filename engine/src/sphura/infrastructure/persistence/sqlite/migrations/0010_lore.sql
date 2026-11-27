-- Lore Studio: a scene index of whole videos, cached lore passages with attribution,
-- and discovered story ideas. Frames are the resumable checkpoint of an indexing job.
CREATE TABLE lore_index (
    asset_id      TEXT PRIMARY KEY REFERENCES media_assets(id) ON DELETE CASCADE,
    project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    game          TEXT,
    fingerprint   TEXT NOT NULL,
    version       INTEGER NOT NULL,
    status        TEXT NOT NULL,  -- done, failed
    data          TEXT NOT NULL,  -- scene count, characters, locations, error
    updated_at    TEXT NOT NULL
);

CREATE TABLE lore_frames (
    asset_id      TEXT NOT NULL REFERENCES media_assets(id) ON DELETE CASCADE,
    fingerprint   TEXT NOT NULL,
    version       INTEGER NOT NULL,
    t_s           REAL NOT NULL,
    data          TEXT NOT NULL,  -- kind, characters, location, objects, what
    PRIMARY KEY (asset_id, fingerprint, version, t_s)
);

CREATE TABLE lore_scenes (
    id            TEXT PRIMARY KEY,
    asset_id      TEXT NOT NULL REFERENCES media_assets(id) ON DELETE CASCADE,
    project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    game          TEXT,
    start_s       REAL NOT NULL,
    end_s         REAL NOT NULL,
    data          TEXT NOT NULL   -- kind, characters, location, objects, what, dialogue, moments
);
CREATE INDEX lore_scenes_asset ON lore_scenes(asset_id, start_s);
CREATE INDEX lore_scenes_game ON lore_scenes(game);

CREATE TABLE lore_pages (
    game          TEXT NOT NULL,
    entity        TEXT NOT NULL,
    source        TEXT NOT NULL,  -- wiki, myth
    status        TEXT NOT NULL,  -- ok, missing, failed
    data          TEXT NOT NULL,  -- page title, url, error
    fetched_at    TEXT NOT NULL,
    PRIMARY KEY (game, entity, source)
);

CREATE TABLE lore_passages (
    id            TEXT PRIMARY KEY,
    game          TEXT NOT NULL,
    entity        TEXT NOT NULL,
    source        TEXT NOT NULL,
    kind          TEXT NOT NULL,  -- canon, quote, trivia, myth, theory
    data          TEXT NOT NULL,  -- text, url, page, section, site, license
    fetched_at    TEXT NOT NULL
);
CREATE INDEX lore_passages_entity ON lore_passages(game, entity);

CREATE TABLE lore_ideas (
    id            TEXT PRIMARY KEY,
    game          TEXT NOT NULL,
    category      TEXT NOT NULL,
    score         REAL NOT NULL,
    data          TEXT NOT NULL,  -- title, pitch, beats, scenes, passages, confidence
    created_at    TEXT NOT NULL
);
CREATE INDEX lore_ideas_game ON lore_ideas(game, score);
