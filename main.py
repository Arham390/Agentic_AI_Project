import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from tools.llm_factory import llm_configured, llm_provider_hint, normalize_llm_env_vars
from workflow.graph import build_graph


def _load_dotenv_if_present() -> None:
	env_path = Path(__file__).resolve().parent / ".env"
	if not env_path.is_file():
		return

	try:
		from dotenv import load_dotenv
	except ImportError:
		# Minimal parser fallback so .env works even without python-dotenv.
		for raw_line in env_path.read_text(encoding="utf-8").splitlines():
			line = raw_line.strip()
			if not line or line.startswith("#") or "=" not in line:
				continue
			key, value = line.split("=", 1)
			key = key.strip()
			value = value.strip()
			if not key:
				continue
			if value and ((value[0] == value[-1]) and value[0] in {"\"", "'"}):
				value = value[1:-1]
			os.environ.setdefault(key, value)
		return

	load_dotenv(env_path)


def _parse_script_to_scenes(script: str) -> List[Dict[str, Any]]:
	heading_re = re.compile(r"^\s*(Scene\s+\d+|INT\.|EXT\.)", re.IGNORECASE)
	dialogue_re = re.compile(
		r"^\s*([A-Z][A-Z0-9_ ]{1,30})(\([^)]+\))?\s*:\s*(.+)$"
	)

	scenes: List[Dict[str, Any]] = []
	current_scene: Dict[str, Any] | None = None

	def ensure_scene() -> Dict[str, Any]:
		nonlocal current_scene
		if current_scene is None:
			current_scene = {
				"scene_id": f"scene_{len(scenes) + 1:02d}",
				"heading": f"Scene {len(scenes) + 1}",
				"dialogues": [],
				"actions": [],
				"raw_lines": [],
			}
			scenes.append(current_scene)
		return current_scene

	for raw_line in script.splitlines():
		line = raw_line.strip()
		if not line:
			continue

		if heading_re.search(line):
			current_scene = {
				"scene_id": f"scene_{len(scenes) + 1:02d}",
				"heading": line,
				"dialogues": [],
				"actions": [],
				"raw_lines": [],
			}
			scenes.append(current_scene)
			continue

		scene = ensure_scene()
		scene["raw_lines"].append(line)

		dialogue_match = dialogue_re.match(line)
		if dialogue_match:
			scene["dialogues"].append(
				{
					"character": dialogue_match.group(1).strip().title(),
					"line": (dialogue_match.group(3) or "").strip(),
				}
			)
		elif line.startswith("(") and line.endswith(")"):
			scene["actions"].append(line)

	return scenes


def script_text_from_scene_manifest(manifest: Dict[str, Any]) -> str:
	"""Rebuild screenplay text from structured scenes so script.txt matches the manifest."""
	scenes = manifest.get("scenes") if isinstance(manifest, dict) else None
	if not isinstance(scenes, list):
		return ""

	lines: List[str] = []
	for scene in scenes:
		if not isinstance(scene, dict):
			continue
		heading = str(scene.get("heading", "")).strip()
		if heading:
			lines.append(heading)

		raw_lines = scene.get("raw_lines")
		if isinstance(raw_lines, list) and raw_lines:
			for raw in raw_lines:
				s = str(raw).strip()
				if s:
					lines.append(s)
			lines.append("")
			continue

		for act in scene.get("actions") or []:
			if isinstance(act, str) and act.strip():
				lines.append(act.strip())

		for d in scene.get("dialogues") or []:
			if not isinstance(d, dict):
				continue
			ch = str(d.get("character", "")).strip()
			ln = str(d.get("line", d.get("text", ""))).strip()
			if ch and ln:
				lines.append(f"{ch.upper()}: {ln}")
			elif ln:
				lines.append(ln)

		lines.append("")

	return "\n".join(lines).strip()


def _build_scene_manifest(result: Dict[str, Any], script: str) -> Dict[str, Any]:
	scenes = _parse_script_to_scenes(script)
	out: Dict[str, Any] = {
		"mode": result.get("mode", "autonomous"),
		"validated": bool(result.get("validated", False)),
		"validation_report": result.get("validation_report", {}),
		"scene_count": len(scenes),
		"scenes": scenes,
		"images": result.get("images", []),
	}
	sr = result.get("stop_reason")
	if sr:
		out["stop_reason"] = sr
	inv = result.get("llm_invocations")
	if inv:
		out["llm_invocations"] = inv
	return out


def _write_json(path: Path, payload: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def run_pipeline(prompt: str, manual_script: str, require_hitl: bool) -> Dict[str, Any]:
	_load_dotenv_if_present()
	normalize_llm_env_vars()
	graph = build_graph()
	initial_state: Dict[str, Any] = {
		"input_prompt": prompt,
		"manual_script": manual_script,
		"require_hitl": require_hitl,
		"approved": False,
		"script": "",
		"validated": False,
		"validation_report": {},
		"characters": [],
		"images": [],
		"scene_manifest_data": {},
		"scene_tasks": [],
		"task_graph_logs": [],
		"audio_tracks": [],
		"video_tracks": [],
		"face_swaps": [],
		"raw_scenes": [],
		"llm_invocations": [],
		"script_repair_count": 0,
	}
	return graph.invoke(initial_state)


def persist_outputs(result: Dict[str, Any]) -> Dict[str, Path]:
	output_dir = Path(__file__).resolve().parent / "outputs"
	output_dir.mkdir(parents=True, exist_ok=True)
	(output_dir / "image_assets").mkdir(parents=True, exist_ok=True)
	(output_dir / "raw_scenes").mkdir(parents=True, exist_ok=True)
	(output_dir / "audio_tracks").mkdir(parents=True, exist_ok=True)
	(output_dir / "intermediate_frames").mkdir(parents=True, exist_ok=True)
	(output_dir / "task_graph_logs").mkdir(parents=True, exist_ok=True)

	script_text = str(result.get("script", ""))
	characters = result.get("characters", [])

	script_path = output_dir / "script.txt"
	character_db_path = output_dir / "character_db.json"
	scene_manifest_path = output_dir / "scene_manifest.json"

	_write_json(
		character_db_path,
		{
			"character_count": len(characters),
			"characters": characters,
		},
	)
	scene_manifest_payload = result.get("scene_manifest_data")
	if not isinstance(scene_manifest_payload, dict) or not scene_manifest_payload.get("scenes"):
		scene_manifest_payload = _build_scene_manifest(result, script_text)

	derived_script = script_text_from_scene_manifest(scene_manifest_payload)
	if derived_script.strip():
		script_text = derived_script.strip() + "\n"

	script_path.write_text(script_text, encoding="utf-8")
	_write_json(scene_manifest_path, scene_manifest_payload)

	return {
		"script": script_path,
		"character_db": character_db_path,
		"scene_manifest": scene_manifest_path,
		"raw_scenes_dir": output_dir / "raw_scenes",
		"audio_tracks_dir": output_dir / "audio_tracks",
		"intermediate_frames_dir": output_dir / "intermediate_frames",
		"task_graph_logs_dir": output_dir / "task_graph_logs",
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run the agentic screenplay pipeline.")
	parser.add_argument("prompt", nargs="*", help="Prompt for screenplay generation")
	parser.add_argument(
		"--prompt",
		dest="prompt_override",
		help="Prompt text. Overrides positional prompt arguments.",
	)
	parser.add_argument(
		"--script-text",
		dest="script_text",
		help="Provide a full manual script directly.",
	)
	parser.add_argument(
		"--script-file",
		dest="script_file",
		help="Path to a UTF-8 text file containing a manual script.",
	)
	parser.add_argument(
		"--hitl",
		action="store_true",
		help="Enable interactive human-in-the-loop approval checkpoint.",
	)
	return parser.parse_args()


def _load_manual_script(args: argparse.Namespace) -> str:
	if args.script_text:
		return str(args.script_text).strip()

	if args.script_file:
		path = Path(args.script_file)
		if not path.exists():
			raise FileNotFoundError(f"Manual script file not found: {path}")
		return path.read_text(encoding="utf-8").strip()

	return ""


def main() -> None:
	_load_dotenv_if_present()
	normalize_llm_env_vars()
	args = parse_args()
	manual_script = _load_manual_script(args)

	prompt = (args.prompt_override or " ".join(args.prompt)).strip()
	if not prompt and not manual_script:
		prompt = "A student builds an AI film crew for a class project."

	result = run_pipeline(prompt=prompt, manual_script=manual_script, require_hitl=args.hitl)
	output_paths = persist_outputs(result)

	print("Pipeline complete.")
	print(f"Mode: {result.get('mode', 'autonomous')}")
	print(f"Validated: {result.get('validated', False)}")
	v = bool(result.get("validated", False))
	rh = bool(result.get("require_hitl", False))
	ua = bool(result.get("approved", False))
	effective_ok = ua or (v and not rh)
	print(f"Approved (effective): {effective_ok}")
	sr = result.get("stop_reason")
	if sr:
		print(f"Stop reason: {sr}")
	print(f"Characters: {len(result.get('characters', []))}")
	print(f"Raw scenes: {len(result.get('raw_scenes', []))}")
	print(f"Audio scene tracks: {len(result.get('audio_tracks', []))}")
	print(f"Video scene tracks: {len(result.get('video_tracks', []))}")
	inv = result.get("llm_invocations") or []
	mode = result.get("mode", "autonomous")
	if inv:
		print("LLM usage (successful calls):")
		for row in inv:
			print(f"  - {row.get('step', '?')}: {row.get('provider', '?')} / {row.get('model', '?')}")
	else:
		print("LLM usage: none (no successful cloud model calls were recorded).")
		if mode == "autonomous":
			if not llm_configured():
				print(
					"Hint: Python does not see GROQ_API_KEY or OPENAI_API_KEY in this process. "
					"Use the same terminal where you set the variable, put keys in a .env file "
					"next to main.py (with python-dotenv installed), or restart the terminal after setx."
				)
			else:
				print(
					f"Hint: A key is set for provider '{llm_provider_hint()}' but every LLM call failed "
					"(wrong model name, network, quota, or package issue). Try GROQ_MODEL=llama-3.3-70b-versatile "
					"or check errors above."
				)
	print(f"Memory commit status: {result.get('memory_commit_status', 'n/a')}")
	print("Output files:")
	print(f"- {output_paths['script']}")
	print(f"- {output_paths['character_db']}")
	print(f"- {output_paths['scene_manifest']}")
	print(f"- {output_paths['raw_scenes_dir']}")
	print(f"- {output_paths['audio_tracks_dir']}")
	print(f"- {output_paths['intermediate_frames_dir']}")
	print(f"- {output_paths['task_graph_logs_dir']}")


if __name__ == "__main__":
	main()
