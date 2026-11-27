-- Voice Intelligence: a pronunciation dictionary and profile per game, plus voice settings.
-- A user's edit or verification always wins over scanned or AI-suggested entries.
CREATE TABLE voice_terms (
    game          TEXT NOT NULL,
    id            TEXT NOT NULL,  -- normalised canonical name
    data          TEXT NOT NULL,  -- canonical, aliases, respell, ipa, notes, origin, confidence
    source        TEXT NOT NULL,  -- builtin, wiki, scan, ai, user
    verified      INTEGER NOT NULL DEFAULT 0,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (game, id)
);

CREATE TABLE voice_profiles (
    game          TEXT PRIMARY KEY,
    data          TEXT NOT NULL,  -- name, default personality, accent, version, last scan
    updated_at    TEXT NOT NULL
);

CREATE TABLE voice_settings (
    scope         TEXT PRIMARY KEY,  -- "global"
    data          TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
