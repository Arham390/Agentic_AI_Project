"""Phase 5 — State snapshot and undo system.

Each pipeline run or edit creates a versioned snapshot.  Callers can revert
to any prior version (both the JSON state and key output files are restored).
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STATE_DIR = _PROJECT_ROOT / "outputs" / "state_versions"
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"

# Files to include in every snapshot.
_SNAPSHOT_FILES = ("script.txt", "character_db.json", "scene_manifest.json")


class StateManager:
    """Append-only state snapshot store with revert support."""

    def __init__(self, state_dir: Optional[Path] = None) -> None:
        self._state_dir = Path(state_dir) if state_dir else _STATE_DIR
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._state_dir / "index.json"

    # ─── internal ─────────────────────────────────────────────────────────────

    def _load_index(self) -> List[Dict[str, Any]]:
        if not self._index_path.exists():
            return []
        try:
            return json.loads(self._index_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_index(self, index: List[Dict[str, Any]]) -> None:
        self._index_path.write_text(
            json.dumps(index, indent=2, default=str), encoding="utf-8"
        )

    def _version_dir(self, version: int) -> Path:
        return self._state_dir / f"v{version:04d}"

    # ─── public API ───────────────────────────────────────────────────────────

    def snapshot(self, state: Dict[str, Any], description: str = "") -> int:
        """Persist *state* and key output files.  Returns the new version number."""
        index = self._load_index()
        version = len(index) + 1
        vdir = self._version_dir(version)
        vdir.mkdir(parents=True, exist_ok=True)

        # Serialise state (skip non-serialisable blobs, best-effort).
        state_path = vdir / "state.json"
        state_path.write_text(
            json.dumps(state, indent=2, default=str), encoding="utf-8"
        )

        # Copy output artefacts into the snapshot directory.
        assets: Dict[str, str] = {}
        for fname in _SNAPSHOT_FILES:
            src = _OUTPUTS_DIR / fname
            if src.exists():
                dst = vdir / fname
                try:
                    shutil.copyfile(src, dst)
                    assets[fname] = fname
                except Exception:
                    pass

        # Also snapshot audio/video track lists as lightweight JSON manifests.
        for key in ("audio_tracks", "video_tracks", "face_swaps", "raw_scenes"):
            val = state.get(key)
            if isinstance(val, list) and val:
                (vdir / f"{key}.json").write_text(
                    json.dumps(val, indent=2, default=str), encoding="utf-8"
                )

        entry: Dict[str, Any] = {
            "version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "description": description[:200],
            "assets": assets,
            "character_count": len(state.get("characters") or []),
            "scene_count": len(
                (state.get("scene_manifest_data") or {}).get("scenes") or []
            ),
            "audio_count": len(state.get("audio_tracks") or []),
            "video_count": len(state.get("video_tracks") or []),
        }
        index.append(entry)
        self._save_index(index)
        return version

    def revert(self, version: int) -> Optional[Dict[str, Any]]:
        """Restore the snapshot for *version*.

        Copies saved assets back to the outputs directory and returns the
        persisted state dict.  Returns ``None`` if the version does not exist.
        """
        index = self._load_index()
        entry = next((e for e in index if e["version"] == version), None)
        if entry is None:
            return None

        vdir = self._version_dir(version)

        # Restore state JSON.
        state_path = vdir / "state.json"
        if not state_path.exists():
            return None
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        # Older snapshots may omit large lists from state.json; merge sidecar JSON if present.
        for aux_key in ("video_tracks", "audio_tracks", "face_swaps", "raw_scenes"):
            aux_path = vdir / f"{aux_key}.json"
            if not aux_path.exists():
                continue
            try:
                aux_val = json.loads(aux_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(aux_val, list) or not aux_val:
                continue
            cur = state.get(aux_key)
            if not isinstance(cur, list) or len(cur) == 0:
                state[aux_key] = aux_val

        # Restore output files.
        for fname in entry.get("assets", {}).keys():
            src = vdir / fname
            dst = _OUTPUTS_DIR / fname
            if src.exists():
                try:
                    shutil.copyfile(src, dst)
                except Exception:
                    pass

        return state

    def load_state(self, version: int) -> Optional[Dict[str, Any]]:
        """Read the state dict for *version* WITHOUT restoring output files.

        Use this when you need the state data (e.g. images, audio_tracks) for
        edit operations but don't want to overwrite the current output files.
        """
        index = self._load_index()
        entry = next((e for e in index if e["version"] == version), None)
        if entry is None:
            return None

        vdir = self._version_dir(version)
        state_path = vdir / "state.json"
        if not state_path.exists():
            return None
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def history(self) -> List[Dict[str, Any]]:
        """Return all version entries (newest last)."""
        return self._load_index()

    def diff_summary(self, v1: int, v2: int) -> Dict[str, Any]:
        """Lightweight diff between two versions."""
        index = self._load_index()
        e1 = next((e for e in index if e["version"] == v1), {})
        e2 = next((e for e in index if e["version"] == v2), {})
        return {
            "from_version": v1,
            "to_version": v2,
            "character_count_delta": e2.get("character_count", 0) - e1.get("character_count", 0),
            "scene_count_delta": e2.get("scene_count", 0) - e1.get("scene_count", 0),
            "audio_count_delta": e2.get("audio_count", 0) - e1.get("audio_count", 0),
            "description_from": e1.get("description", ""),
            "description_to": e2.get("description", ""),
        }
