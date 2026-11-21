CREATE TABLE feedback (
    candidate_id  TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL,
    good          INTEGER NOT NULL,
    traits        TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX idx_feedback_project ON feedback(project_id);
