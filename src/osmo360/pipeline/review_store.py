from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

DECISIONS = {"approved", "reprocess", "rejected", "reprocessed"}
REASONS = {
    "blurry": "画面看不清", "sync": "左右对不上", "gripper": "夹爪识别错了",
    "pose_jump": "定位突然跳动", "occlusion": "遮挡太多", "incomplete": "视频不完整",
    "three_d": "3D和视频对不上", "missing_3d": "没有3D审核画面", "other": "其他",
}
SEGMENT_LABELS = {
    "pick": "拿起物体", "place": "放下物体", "move": "移动物体",
    "sort": "分类整理", "idle": "等待或无动作", "invalid": "无效片段", "other": "其他",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_fingerprint(directory: Path) -> str:
    digest = hashlib.sha256()
    instaumi = directory.name.startswith("instaumi_")
    names = (
        ("dataset.h5", "video/Left.mp4", "video/Right.mp4", "processed/review.json")
        if instaumi else
        ("metrics.json", "tag_detections.jsonl", "gripper_stats.csv", "rgb_samples.npz",
         "review_bundle/timeline.json", "review_bundle/front-video.mp4",
         "review_bundle/scene.html", "review_bundle/visualization.json")
    )
    for name in names:
        path = directory / name
        if not path.is_file():
            digest.update(f"missing:{name}".encode()); continue
        digest.update(name.encode())
        if instaumi and path.suffix == ".mp4":
            stat = path.stat()
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
        else:
            digest.update(sha256(path).encode())
    return digest.hexdigest()


def summarize_pair(directory: Path) -> dict[str, Any]:
    if directory.name.startswith("instaumi_"):
        review_path = directory / "processed" / "review.json"
        review = json.loads(review_path.read_text()) if review_path.is_file() else {}
        videos_ready = all((directory / "video" / name).is_file() for name in ("Left.mp4", "Right.mp4"))
        return {
            "auto_status": "pass_candidate" if videos_ready else "needs_review",
            "complete": videos_ready and (directory / "dataset.h5").is_file(),
            "duration_s": float(review.get("duration_s", 0)), "wall_s": 0.0,
            "rgb_samples": max(0, int(float(review.get("duration_s", 0)))),
            "tag_frames": 0, "tag_usable_ratio": 0.0, "tag_count_average": 0,
            "gripper_frames": 0, "gripper_candidate_ratio": 0.0,
            "yellow_pixels_average": 0, "fallbacks": {},
            "three_d_generated": False, "three_d_ready": False, "three_d_url": None,
            "aligned_video_ready": videos_ready, "review_ready": videos_ready,
            "video_fps": float(review.get("video_fps", 60.0)),
            "sync": review.get("sync", {}),
        }
    metrics_path = directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    tag_rows = [json.loads(line) for line in (directory / "tag_detections.jsonl").read_text().splitlines() if line] if (directory / "tag_detections.jsonl").is_file() else []
    candidate_counts: list[int] = []; yellow_pixels: list[int] = []
    if (directory / "gripper_stats.csv").is_file():
        with (directory / "gripper_stats.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                candidate_counts.append(int(row["candidate_count"])); yellow_pixels.append(int(row["yellow_pixels"]))
    rgb_samples = 0
    try:
        import numpy as np
        with np.load(directory / "rgb_samples.npz") as arrays:
            rgb_samples = min(len(arrays["left"]), len(arrays["right"]))
    except (FileNotFoundError, KeyError, ValueError):
        pass
    tag_counts = [len(row.get("ids", [])) for row in tag_rows]
    tag_usable = sum(x >= 2 for x in tag_counts) / len(tag_counts) if tag_counts else 0.0
    candidate_usable = sum(x > 0 for x in candidate_counts) / len(candidate_counts) if candidate_counts else 0.0
    complete = all((directory / name).is_file() for name in ("metrics.json", "tag_detections.jsonl", "gripper_stats.csv", "rgb_samples.npz"))
    bundle = directory / "review_bundle"; visualization_path = bundle / "visualization.json"; visualization = {}
    if visualization_path.is_file():
        try: visualization = json.loads(visualization_path.read_text())
        except json.JSONDecodeError: pass
    three_d_generated = all((bundle / name).is_file() for name in ("timeline.json", "front-video.mp4", "scene.html"))
    three_d_url = str(visualization.get("view_url", "")).strip() or None
    three_d_ready = three_d_generated and bool(three_d_url)
    return {
        "auto_status": "pass_candidate" if complete and three_d_ready and tag_usable >= .98 and candidate_usable >= .90 else "needs_review",
        "complete": complete, "duration_s": float(metrics.get("input_duration_seconds_per_role", 0)),
        "wall_s": float(metrics.get("wall_clock_seconds", 0)), "rgb_samples": rgb_samples,
        "tag_frames": len(tag_rows), "tag_usable_ratio": tag_usable,
        "tag_count_average": sum(tag_counts) / len(tag_counts) if tag_counts else 0,
        "gripper_frames": len(candidate_counts), "gripper_candidate_ratio": candidate_usable,
        "yellow_pixels_average": sum(yellow_pixels) / len(yellow_pixels) if yellow_pixels else 0,
        "fallbacks": {role: int(v.get("counts", {}).get("fallback", 0)) for role, v in metrics.get("roles", {}).items()},
        "three_d_generated": three_d_generated, "three_d_ready": three_d_ready,
        "three_d_url": three_d_url, "aligned_video_ready": False,
        "review_ready": three_d_ready,
    }


class ReviewStore:
    def __init__(
        self, dataset_root: Path, database: Path | None = None,
        state_root: Path | None = None,
    ) -> None:
        self.dataset_root = dataset_root.resolve(); self.preprocess_root = self.dataset_root / "preprocess"
        review_root = (state_root or self.dataset_root / "review").resolve()
        review_root.mkdir(parents=True, exist_ok=True)
        self.database = (database or review_root / "reviews.sqlite").resolve(); self.snapshot = review_root / "reviews.snapshot.sqlite"
        self.queue_path = review_root / "reprocess_queue.json"; self.export_path = review_root / "approved_segments.json"
        self.alignment_path = review_root / "alignment_adjustments.json"
        self.keyframes_path = review_root / "keyframes.json"
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database); connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON"); connection.execute("PRAGMA journal_mode=WAL"); return connection

    def _initialize(self) -> None:
        with self.connect() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS items(pair_id TEXT PRIMARY KEY,source_dir TEXT NOT NULL,data_hash TEXT NOT NULL,auto_status TEXT NOT NULL,metrics_json TEXT NOT NULL,scanned_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS review_events(id INTEGER PRIMARY KEY AUTOINCREMENT,pair_id TEXT NOT NULL REFERENCES items(pair_id),data_hash TEXT NOT NULL,decision TEXT NOT NULL,reasons_json TEXT NOT NULL,notes TEXT NOT NULL,reviewer TEXT NOT NULL,created_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS review_pair_created ON review_events(pair_id,created_at DESC,id DESC);
            CREATE TABLE IF NOT EXISTS reprocess_queue(pair_id TEXT PRIMARY KEY REFERENCES items(pair_id),data_hash TEXT NOT NULL,review_event_id INTEGER NOT NULL REFERENCES review_events(id),status TEXT NOT NULL,updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS segments(id INTEGER PRIMARY KEY AUTOINCREMENT,pair_id TEXT NOT NULL REFERENCES items(pair_id),data_hash TEXT NOT NULL,start_s REAL NOT NULL,end_s REAL NOT NULL,label TEXT NOT NULL,success INTEGER NOT NULL,notes TEXT NOT NULL,reviewer TEXT NOT NULL,created_at TEXT NOT NULL,superseded_by INTEGER REFERENCES segments(id));
            CREATE TABLE IF NOT EXISTS segment_reviews(id INTEGER PRIMARY KEY AUTOINCREMENT,segment_id INTEGER NOT NULL REFERENCES segments(id),decision TEXT NOT NULL,reasons_json TEXT NOT NULL,notes TEXT NOT NULL,reviewer TEXT NOT NULL,created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS alignment_adjustments(id INTEGER PRIMARY KEY AUTOINCREMENT,pair_id TEXT NOT NULL REFERENCES items(pair_id),data_hash TEXT NOT NULL,right_time_offset_s REAL NOT NULL,reviewer TEXT NOT NULL,notes TEXT NOT NULL,created_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS alignment_pair_created ON alignment_adjustments(pair_id,id DESC);
            CREATE TABLE IF NOT EXISTS keyframes(id TEXT PRIMARY KEY,pair_id TEXT NOT NULL REFERENCES items(pair_id),data_hash TEXT NOT NULL,frame INTEGER NOT NULL,time_sec REAL NOT NULL,label TEXT NOT NULL,reviewer TEXT NOT NULL,created_at TEXT NOT NULL,deleted_at TEXT);
            CREATE INDEX IF NOT EXISTS keyframe_pair_time ON keyframes(pair_id,time_sec,id);
            """)

    def scan(self) -> list[dict[str, Any]]:
        directories = []
        if self.preprocess_root.is_dir():
            directories.extend(sorted(self.preprocess_root.glob("pair-*")))
        directories.extend(sorted(self.dataset_root.glob("instaumi_[0-9][0-9][0-9][0-9][0-9][0-9]")))
        if not directories:
            raise FileNotFoundError(
                f"no preprocess/pair-* or instaumi_xxxxxx datasets found: {self.dataset_root}"
            )
        with self.connect() as c:
            for directory in directories:
                if not directory.is_dir(): continue
                metrics = summarize_pair(directory)
                c.execute("""INSERT INTO items VALUES(?,?,?,?,?,?) ON CONFLICT(pair_id) DO UPDATE SET source_dir=excluded.source_dir,data_hash=excluded.data_hash,auto_status=excluded.auto_status,metrics_json=excluded.metrics_json,scanned_at=excluded.scanned_at""",(directory.name,str(directory),pair_fingerprint(directory),metrics["auto_status"],json.dumps(metrics),utc_now()))
        return self.list_items()

    def list_items(self) -> list[dict[str, Any]]:
        query="""SELECT i.*,r.id review_id,r.data_hash review_hash,r.decision,r.reasons_json,r.notes,r.reviewer,r.created_at FROM items i LEFT JOIN review_events r ON r.id=(SELECT id FROM review_events WHERE pair_id=i.pair_id ORDER BY id DESC LIMIT 1) ORDER BY i.pair_id"""
        with self.connect() as c: rows=c.execute(query).fetchall()
        result=[]
        for row in rows:
            item=dict(row);item["metrics"]=json.loads(item.pop("metrics_json"));item["reasons"]=json.loads(item.pop("reasons_json")) if item.get("reasons_json") else []
            item["stale_review"]=bool(item.get("review_hash") and item["review_hash"]!=item["data_hash"])
            if item["stale_review"]: item["decision"]=None
            result.append(item)
        return result

    def get_item(self,pair_id:str)->dict[str,Any]:
        for item in self.list_items():
            if item["pair_id"]==pair_id:return item
        raise KeyError(pair_id)

    def history(self,pair_id:str)->list[dict[str,Any]]:
        with self.connect() as c: rows=c.execute("SELECT * FROM review_events WHERE pair_id=? ORDER BY id DESC",(pair_id,)).fetchall()
        result=[]
        for row in rows:
            event=dict(row);event["reasons"]=json.loads(event.pop("reasons_json"));result.append(event)
        return result

    def get_alignment(self, pair_id: str) -> dict[str, Any]:
        item = self.get_item(pair_id)
        with self.connect() as c:
            row = c.execute(
                "SELECT * FROM alignment_adjustments WHERE pair_id=? ORDER BY id DESC LIMIT 1",
                (pair_id,),
            ).fetchone()
        if row is None:
            return {"pair_id": pair_id, "right_time_offset_s": 0.0, "saved": False, "stale": False}
        result = dict(row)
        result["saved"] = True
        result["stale"] = result["data_hash"] != item["data_hash"]
        if result["stale"]:
            result["right_time_offset_s"] = 0.0
        return result

    def save_alignment(
        self, pair_id: str, *, right_time_offset_s: float,
        reviewer: str, notes: str = "",
    ) -> dict[str, Any]:
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        if not -30.0 <= right_time_offset_s <= 30.0:
            raise ValueError("manual alignment offset must be within ±30 seconds")
        item = self.get_item(pair_id)
        with self.connect() as c:
            cursor = c.execute(
                "INSERT INTO alignment_adjustments(pair_id,data_hash,right_time_offset_s,reviewer,notes,created_at) VALUES(?,?,?,?,?,?)",
                (pair_id, item["data_hash"], float(right_time_offset_s), reviewer.strip(), notes.strip(), utc_now()),
            )
            row = c.execute(
                "SELECT * FROM alignment_adjustments WHERE id=?", (cursor.lastrowid,)
            ).fetchone()
        self._export()
        return {**dict(row), "saved": True, "stale": False}

    def add_review(self,pair_id:str,*,decision:str,reasons:list[str],notes:str,reviewer:str)->dict[str,Any]:
        if decision not in DECISIONS: raise ValueError("invalid decision")
        if set(reasons)-REASONS.keys(): raise ValueError("invalid reasons")
        if not reviewer.strip(): raise ValueError("reviewer is required")
        if decision in {"reprocess","rejected"} and not reasons: raise ValueError("a reason is required")
        item=self.get_item(pair_id)
        if decision=="approved" and not item["metrics"].get("review_ready"): raise ValueError("必须先生成可审核的对齐视频或3D同步画面，才能通过")
        now=utc_now()
        with self.connect() as c:
            cur=c.execute("INSERT INTO review_events(pair_id,data_hash,decision,reasons_json,notes,reviewer,created_at) VALUES(?,?,?,?,?,?,?)",(pair_id,item["data_hash"],decision,json.dumps(reasons),notes.strip(),reviewer.strip(),now));event_id=int(cur.lastrowid)
            if decision=="reprocess": c.execute("INSERT INTO reprocess_queue VALUES(?,?,?,?,?) ON CONFLICT(pair_id) DO UPDATE SET data_hash=excluded.data_hash,review_event_id=excluded.review_event_id,status='queued',updated_at=excluded.updated_at",(pair_id,item["data_hash"],event_id,"queued",now))
            else: c.execute("UPDATE reprocess_queue SET status='cancelled',updated_at=? WHERE pair_id=? AND status='queued'",(now,pair_id))
        self._export();return self.history(pair_id)[0]

    def list_keyframes(self, pair_id: str) -> list[dict[str, Any]]:
        item = self.get_item(pair_id)
        with self.connect() as c:
            rows = c.execute(
                "SELECT id,frame,time_sec,label,created_at FROM keyframes "
                "WHERE pair_id=? AND data_hash=? AND deleted_at IS NULL "
                "ORDER BY time_sec,id",
                (pair_id, item["data_hash"]),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_keyframe(
        self, pair_id: str, *, time_sec: float, label: str, reviewer: str,
    ) -> dict[str, Any]:
        if label not in {"useful_start", "useful_end"}:
            raise ValueError("invalid keyframe label")
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        item = self.get_item(pair_id)
        if item.get("decision") != "approved" or item.get("stale_review"):
            raise ValueError("请先确认整条数据能用")
        duration = float(item["metrics"].get("duration_s", 0))
        if not 0 <= time_sec <= duration:
            raise ValueError("keyframe time is outside the video")
        fps = float(item["metrics"].get("video_fps", 30.0))
        created_at = utc_now().replace("+00:00", "Z")
        row = {
            "id": str(uuid4()), "pair_id": pair_id,
            "data_hash": item["data_hash"], "frame": round(time_sec * fps),
            "time_sec": round(float(time_sec), 3), "label": label,
            "reviewer": reviewer.strip(), "created_at": created_at,
        }
        with self.connect() as c:
            c.execute(
                "INSERT INTO keyframes(id,pair_id,data_hash,frame,time_sec,label,reviewer,created_at,deleted_at) "
                "VALUES(:id,:pair_id,:data_hash,:frame,:time_sec,:label,:reviewer,:created_at,NULL)",
                row,
            )
        self._export()
        return {key: row[key] for key in ("id", "frame", "time_sec", "label", "created_at")}

    def delete_keyframe(self, keyframe_id: str) -> None:
        with self.connect() as c:
            cursor = c.execute(
                "UPDATE keyframes SET deleted_at=? WHERE id=? AND deleted_at IS NULL",
                (utc_now(), keyframe_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(keyframe_id)
        self._export()

    def add_segment(self,pair_id:str,*,start_s:float,end_s:float,label:str,success:bool,notes:str,reviewer:str)->dict[str,Any]:
        item=self.get_item(pair_id);duration=float(item["metrics"]["duration_s"])
        if label not in SEGMENT_LABELS: raise ValueError("invalid segment label")
        if not reviewer.strip(): raise ValueError("reviewer is required")
        if not (0<=start_s<end_s<=duration): raise ValueError("invalid segment range")
        if end_s-start_s<0.5: raise ValueError("segment must be at least 0.5 seconds")
        with self.connect() as c:
            cur=c.execute("INSERT INTO segments(pair_id,data_hash,start_s,end_s,label,success,notes,reviewer,created_at,superseded_by) VALUES(?,?,?,?,?,?,?,?,?,NULL)",(pair_id,item["data_hash"],start_s,end_s,label,int(success),notes.strip(),reviewer.strip(),utc_now()))
            row=c.execute("SELECT * FROM segments WHERE id=?",(cur.lastrowid,)).fetchone()
        self._export();return dict(row)

    def list_segments(self,pair_id:str)->list[dict[str,Any]]:
        with self.connect() as c:
            rows=c.execute("""SELECT s.*,r.decision,r.reasons_json review_reasons,r.notes review_notes,r.reviewer review_reviewer,r.created_at reviewed_at FROM segments s LEFT JOIN segment_reviews r ON r.id=(SELECT id FROM segment_reviews WHERE segment_id=s.id ORDER BY id DESC LIMIT 1) WHERE s.pair_id=? AND s.superseded_by IS NULL ORDER BY s.start_s""",(pair_id,)).fetchall()
        result=[]
        for row in rows:
            item=dict(row);item["review_reasons"]=json.loads(item["review_reasons"]) if item.get("review_reasons") else [];result.append(item)
        return result

    def review_segment(self,segment_id:int,*,decision:str,reasons:list[str],notes:str,reviewer:str)->dict[str,Any]:
        if decision not in DECISIONS: raise ValueError("invalid decision")
        if set(reasons)-REASONS.keys(): raise ValueError("invalid reasons")
        if not reviewer.strip(): raise ValueError("reviewer is required")
        with self.connect() as c:
            segment=c.execute("SELECT s.*,i.metrics_json FROM segments s JOIN items i USING(pair_id) WHERE s.id=?",(segment_id,)).fetchone()
            if segment is None: raise KeyError(segment_id)
            if decision=="approved" and not json.loads(segment["metrics_json"]).get("review_ready"): raise ValueError("必须先完成对齐视频或3D同步审核，片段才能通过")
            cur=c.execute("INSERT INTO segment_reviews(segment_id,decision,reasons_json,notes,reviewer,created_at) VALUES(?,?,?,?,?,?)",(segment_id,decision,json.dumps(reasons),notes.strip(),reviewer.strip(),utc_now()))
            row=c.execute("SELECT * FROM segment_reviews WHERE id=?",(cur.lastrowid,)).fetchone()
        self._export();return dict(row)

    def _export(self)->None:
        with self.connect() as c:
            queue_rows=c.execute("""SELECT q.*,i.source_dir,e.reasons_json,e.notes,e.reviewer
                FROM reprocess_queue q JOIN items i USING(pair_id)
                JOIN review_events e ON e.id=q.review_event_id
                WHERE q.status='queued' ORDER BY q.updated_at""").fetchall()
            approved=[dict(row) for row in c.execute("""SELECT s.*,r.id review_id,r.reviewer review_reviewer,r.created_at reviewed_at FROM segments s JOIN segment_reviews r ON r.id=(SELECT id FROM segment_reviews WHERE segment_id=s.id ORDER BY id DESC LIMIT 1) WHERE s.superseded_by IS NULL AND r.decision='approved' ORDER BY s.pair_id,s.start_s""")]
            alignments=[dict(row) for row in c.execute("""SELECT a.* FROM alignment_adjustments a JOIN items i USING(pair_id) WHERE a.id=(SELECT id FROM alignment_adjustments WHERE pair_id=a.pair_id ORDER BY id DESC LIMIT 1) AND a.data_hash=i.data_hash ORDER BY a.pair_id""")]
            keyframe_rows=[dict(row) for row in c.execute("""SELECT k.pair_id,k.id,k.frame,k.time_sec,k.label,k.created_at FROM keyframes k JOIN items i USING(pair_id) WHERE k.data_hash=i.data_hash AND k.deleted_at IS NULL ORDER BY k.pair_id,k.time_sec,k.id""")]
        queue=[]
        for row in queue_rows:
            item=dict(row);item["reasons"]=json.loads(item.pop("reasons_json"));queue.append(item)
        self.queue_path.write_text(json.dumps({"schema_version":"reprocess-queue/v1","items":queue},ensure_ascii=False,indent=2)+"\n")
        self.export_path.write_text(json.dumps({"schema_version":"approved-segments/v1","items":approved},ensure_ascii=False,indent=2)+"\n")
        self.alignment_path.write_text(json.dumps({
            "schema_version": "instaumi-manual-alignment/v1",
            "mapping": "right_video_time_s = left_video_time_s + right_time_offset_s",
            "items": alignments,
        }, ensure_ascii=False, indent=2) + "\n")
        keyframes: dict[str, list[dict[str, Any]]] = {}
        for row in keyframe_rows:
            pair_id = row.pop("pair_id")
            keyframes.setdefault(pair_id, []).append(row)
        self.keyframes_path.write_text(
            json.dumps(keyframes, ensure_ascii=False, indent=2) + "\n"
        )
        with self.connect() as source,sqlite3.connect(self.snapshot) as target: source.backup(target)
