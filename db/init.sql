-- Schema for Blood Cell VisionOps analysis results.
-- Applied automatically by docker-compose and by bloodcell.db.init_db().

CREATE TABLE IF NOT EXISTS blood_cell_detections (
    id                 BIGSERIAL PRIMARY KEY,
    filename           TEXT        NOT NULL,
    image_width        INTEGER     NOT NULL,
    image_height       INTEGER     NOT NULL,
    model_name         TEXT        NOT NULL,
    inference_ms       REAL        NOT NULL,
    total_detections   INTEGER     NOT NULL,
    rbc_count          INTEGER     NOT NULL,
    wbc_count          INTEGER     NOT NULL,
    platelet_count     INTEGER     NOT NULL,
    qc_status          TEXT        NOT NULL,
    quality_control    JSONB       NOT NULL,
    detections         JSONB       NOT NULL,
    ai_report          TEXT,
    report_source      TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_bcd_created_at ON blood_cell_detections (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_bcd_filename   ON blood_cell_detections (filename);
CREATE INDEX IF NOT EXISTS idx_bcd_qc_status  ON blood_cell_detections (qc_status);

-- v1.1: keep a compressed copy of each analysed image so any upload can be reviewed later.
ALTER TABLE blood_cell_detections ADD COLUMN IF NOT EXISTS image_jpeg BYTEA;
