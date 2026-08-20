from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MSMMR_ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\MSMMR")
EMMR_ROOT = Path(r"D:\AAAAAAAAAA_emotion\Code\EMMR")

CATALOG_PATH = MSMMR_ROOT / "outputs" / "inference" / "mtg" / "mtg_music_catalog.npz"
MSMMR_TEXT_CHECKPOINT = MSMMR_ROOT / "outputs" / "text" / "va" / "xlm_roberta_base" / "seed42_20260806-110723" / "best.pt"
FIRST_PAPER_TEXT_CHECKPOINT = EMMR_ROOT / "backend" / "data" / "last.ckpt"
FIRST_PAPER_MUSIC_DB = EMMR_ROOT / "backend" / "data" / "semantic_db.json"
TRACK_METADATA_PATH = FIRST_PAPER_MUSIC_DB
SESSION_LOG_PATH = PROJECT_ROOT / "outputs" / "user_sessions.jsonl"
FIRST_FEEDBACK_LOG_PATH = PROJECT_ROOT / "outputs" / "first_paper_feedback.jsonl"
SC_CAP_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

DEVICE = "cuda"
