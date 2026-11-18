#!/usr/bin/env python3
"""Convert a pyannote database manifest into DiariZen/Kaldi-style data files.

The generated data tree contains per-dataset manifests and combined train/dev/test
manifests. Combined recording IDs are namespaced with the dataset name so datasets
with overlapping recording IDs remain safe to load together.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


DEFAULT_DATABASE = Path(
    "/dataC/jianyou/pyannote/tutorials/AMI-diarization-setup/pyannote/"
    "database_DiariZen.yml"
)
DEFAULT_OUTPUT = Path(
    "/dataC/jianyou/DiariZen/recipes/diar_ssl/data/DiariZen_all"
)
DEFAULT_RAMC = Path(
    "/dataE/jianyou/Diarization/datasets/MagicData-RAMC/processed"
)
DEFAULT_NOTSOFAR = Path(
    "/dataE/jianyou/Diarization/datasets/NOTSOFAR/processed"
)

SPLIT_ALIASES = {"train": "train", "development": "dev", "test": "test"}


@dataclass(frozen=True)
class Recording:
    recording_id: str
    audio: str


@dataclass
class PreparedSplit:
    dataset: str
    split: str
    recordings: list[Recording]
    rttm_lines: list[str]
    uem_lines: list[str]
    source: dict[str, str]

    def validate(self) -> None:
        if not self.recordings:
            raise ValueError(f"{self.dataset}/{self.split}: no recordings")

        recording_ids = [item.recording_id for item in self.recordings]
        duplicates = sorted(
            key for key, count in Counter(recording_ids).items() if count > 1
        )
        if duplicates:
            raise ValueError(
                f"{self.dataset}/{self.split}: duplicate recording IDs: "
                f"{duplicates[:10]}"
            )
        known = set(recording_ids)

        uem_ids: list[str] = []
        for line_number, line in enumerate(self.uem_lines, 1):
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(
                    f"{self.dataset}/{self.split}: malformed UEM line "
                    f"{line_number}: {line!r}"
                )
            uem_ids.append(fields[0])
            float(fields[2])
            float(fields[3])

        unknown_uem = sorted(set(uem_ids) - known)
        missing_uem = sorted(known - set(uem_ids))
        if unknown_uem or missing_uem:
            raise ValueError(
                f"{self.dataset}/{self.split}: UEM/recording mismatch; "
                f"unknown={unknown_uem[:10]}, missing={missing_uem[:10]}"
            )

        rttm_ids: set[str] = set()
        for line_number, line in enumerate(self.rttm_lines, 1):
            fields = line.split()
            if len(fields) < 8 or fields[0] != "SPEAKER":
                raise ValueError(
                    f"{self.dataset}/{self.split}: malformed RTTM line "
                    f"{line_number}: {line!r}"
                )
            float(fields[3])
            duration = float(fields[4])
            if duration <= 0:
                raise ValueError(
                    f"{self.dataset}/{self.split}: non-positive RTTM duration "
                    f"on line {line_number}: {line!r}"
                )
            rttm_ids.add(fields[1])

        unknown_rttm = sorted(rttm_ids - known)
        if unknown_rttm:
            raise ValueError(
                f"{self.dataset}/{self.split}: RTTM contains unknown recordings: "
                f"{unknown_rttm[:10]}"
            )

        for item in self.recordings:
            if any(char.isspace() for char in item.recording_id):
                raise ValueError(
                    f"{self.dataset}/{self.split}: recording ID contains whitespace: "
                    f"{item.recording_id!r}"
                )
            if any(char.isspace() for char in item.audio):
                raise ValueError(
                    f"{self.dataset}/{self.split}: audio path contains whitespace "
                    f"and cannot be represented safely in wav.scp: {item.audio!r}"
                )
            if not Path(item.audio).is_file():
                raise FileNotFoundError(
                    f"{self.dataset}/{self.split}: missing audio: {item.audio}"
                )

    def summary(self) -> dict[str, Any]:
        unique_speakers = {
            line.split()[7] for line in self.rttm_lines if line.strip()
        }
        annotated_seconds = sum(
            float(fields[3]) - float(fields[2])
            for fields in (line.split() for line in self.uem_lines)
        )
        return {
            "recordings": len(self.recordings),
            "rttm_segments": len(self.rttm_lines),
            "unique_speaker_labels": len(unique_speakers),
            "annotated_hours": annotated_seconds / 3600.0,
            "source": self.source,
        }


def nonempty_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_from_manifest(manifest: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest.parent / path
    return path.resolve()


def read_uris(path: Path) -> list[str]:
    uris: list[str] = []
    for line_number, line in enumerate(nonempty_lines(path), 1):
        if line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 1:
            raise ValueError(f"Malformed URI list line {path}:{line_number}: {line!r}")
        uris.append(fields[0])
    if not uris:
        raise ValueError(f"No URIs in {path}")
    return uris


def replace_uri(template: str, uri: str, manifest: Path) -> Path:
    if "{uri}" not in template:
        raise ValueError(f"Template does not contain {{uri}}: {template}")
    return resolve_from_manifest(manifest, template.replace("{uri}", uri))


def prepare_pyannote_split(
    manifest_path: Path,
    manifest_data: dict[str, Any],
    dataset: str,
    protocol_name: str,
    source_split: str,
) -> PreparedSplit:
    try:
        protocol = manifest_data["Protocols"][dataset]["SpeakerDiarization"][protocol_name]
        split_config = protocol[source_split]
        audio_template = manifest_data["Databases"][dataset]
    except KeyError as error:
        raise KeyError(
            f"Missing pyannote configuration for "
            f"{dataset}.SpeakerDiarization.{protocol_name}/{source_split}"
        ) from error

    uri_list = resolve_from_manifest(manifest_path, str(split_config["uri"]))
    uris = read_uris(uri_list)
    recordings: list[Recording] = []
    rttm_lines: list[str] = []
    uem_lines: list[str] = []

    for uri in uris:
        audio_path = replace_uri(str(audio_template), uri, manifest_path)
        rttm_path = replace_uri(str(split_config["annotation"]), uri, manifest_path)
        uem_path = replace_uri(str(split_config["annotated"]), uri, manifest_path)
        recordings.append(Recording(uri, str(audio_path)))

        uri_rttm = nonempty_lines(rttm_path)
        for line in uri_rttm:
            fields = line.split()
            if len(fields) < 2 or fields[1] != uri:
                raise ValueError(
                    f"RTTM ID does not match URI {uri!r} in {rttm_path}: {line!r}"
                )
        rttm_lines.extend(uri_rttm)

        uri_uem = nonempty_lines(uem_path)
        for line in uri_uem:
            fields = line.split()
            if not fields or fields[0] != uri:
                raise ValueError(
                    f"UEM ID does not match URI {uri!r} in {uem_path}: {line!r}"
                )
        uem_lines.extend(uri_uem)

    result = PreparedSplit(
        dataset=dataset,
        split=SPLIT_ALIASES[source_split],
        recordings=recordings,
        rttm_lines=rttm_lines,
        uem_lines=uem_lines,
        source={
            "kind": "pyannote",
            "database_yml": str(manifest_path),
            "protocol": f"{dataset}.SpeakerDiarization.{protocol_name}",
            "protocol_split": source_split,
            "uri_list": str(uri_list),
        },
    )
    result.validate()
    return result


def parse_wav_scp(path: Path) -> list[Recording]:
    recordings: list[Recording] = []
    for line_number, line in enumerate(nonempty_lines(path), 1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"Malformed wav.scp line {path}:{line_number}: {line!r}")
        recording_id, audio = fields
        if audio.endswith("|"):
            raise ValueError(
                f"Command-style wav.scp is not supported for consolidation: {line!r}"
            )
        audio_path = Path(audio)
        if not audio_path.is_absolute():
            audio_path = (path.parent / audio_path).resolve()
        recordings.append(Recording(recording_id, str(audio_path)))
    return recordings


def prepare_existing_split(dataset: str, processed_root: Path, split: str) -> PreparedSplit:
    source_dir = processed_root / split
    result = PreparedSplit(
        dataset=dataset,
        split=split,
        recordings=parse_wav_scp(source_dir / "wav.scp"),
        rttm_lines=nonempty_lines(source_dir / "rttm"),
        uem_lines=nonempty_lines(source_dir / "all.uem"),
        source={"kind": "existing_diarizen", "directory": str(source_dir.resolve())},
    )
    result.validate()
    return result


def discover_pyannote_splits(
    manifest_path: Path, manifest_data: dict[str, Any]
) -> list[PreparedSplit]:
    try:
        combined = manifest_data["Protocols"]["X"]["SpeakerDiarization"]["combined"]
    except KeyError as error:
        raise KeyError("Missing X.SpeakerDiarization.combined protocol") from error

    discovered: list[PreparedSplit] = []
    dataset_protocols: list[tuple[str, str]] = []
    seen_protocols: set[tuple[str, str]] = set()

    for source_split in ("train", "development"):
        for protocol_reference, configured_splits in combined[source_split].items():
            marker = ".SpeakerDiarization."
            if marker not in protocol_reference:
                raise ValueError(f"Unexpected protocol reference: {protocol_reference}")
            dataset, protocol_name = protocol_reference.split(marker, 1)
            key = (dataset, protocol_name)
            if key not in seen_protocols:
                dataset_protocols.append(key)
                seen_protocols.add(key)
            for configured_split in configured_splits:
                if configured_split != source_split:
                    raise ValueError(
                        f"Unsupported combined split mapping for {protocol_reference}: "
                        f"{source_split} -> {configured_split}"
                    )
                discovered.append(
                    prepare_pyannote_split(
                        manifest_path,
                        manifest_data,
                        dataset,
                        protocol_name,
                        configured_split,
                    )
                )

    for dataset, protocol_name in dataset_protocols:
        protocol = manifest_data["Protocols"][dataset]["SpeakerDiarization"][protocol_name]
        if "test" in protocol:
            discovered.append(
                prepare_pyannote_split(
                    manifest_path,
                    manifest_data,
                    dataset,
                    protocol_name,
                    "test",
                )
            )

    return discovered


def write_lines(path: Path, lines: Iterable[str]) -> None:
    materialized = list(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("\n".join(materialized) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_split(output_dir: Path, split: PreparedSplit) -> None:
    write_lines(
        output_dir / "wav.scp",
        (f"{item.recording_id} {item.audio}" for item in split.recordings),
    )
    write_lines(output_dir / "rttm", split.rttm_lines)
    write_lines(output_dir / "all.uem", split.uem_lines)


def namespace_rttm(lines: Iterable[str], prefix: str) -> Iterable[str]:
    for line in lines:
        fields = line.split()
        fields[1] = f"{prefix}__{fields[1]}"
        yield " ".join(fields)


def namespace_uem(lines: Iterable[str], prefix: str) -> Iterable[str]:
    for line in lines:
        fields = line.split()
        fields[0] = f"{prefix}__{fields[0]}"
        yield " ".join(fields)


def combine_splits(splits: list[PreparedSplit], split_name: str) -> PreparedSplit:
    selected = [item for item in splits if item.split == split_name]
    recordings: list[Recording] = []
    rttm_lines: list[str] = []
    uem_lines: list[str] = []
    for item in selected:
        prefix = item.dataset
        recordings.extend(
            Recording(f"{prefix}__{recording.recording_id}", recording.audio)
            for recording in item.recordings
        )
        rttm_lines.extend(namespace_rttm(item.rttm_lines, prefix))
        uem_lines.extend(namespace_uem(item.uem_lines, prefix))

    combined = PreparedSplit(
        dataset="combined",
        split=split_name,
        recordings=recordings,
        rttm_lines=rttm_lines,
        uem_lines=uem_lines,
        source={
            "kind": "combined",
            "datasets": [item.dataset for item in selected],
            "recording_id_format": "<dataset>__<original_recording_id>",
        },
    )
    combined.validate()
    return combined


def verify_audio(splits: list[PreparedSplit]) -> dict[str, Any]:
    import soundfile as sf

    unique_audio = {
        recording.audio
        for split in splits
        for recording in split.recordings
    }
    sample_rates: Counter[int] = Counter()
    channel_counts: Counter[int] = Counter()
    formats: Counter[str] = Counter()
    failures: list[str] = []

    for index, audio in enumerate(sorted(unique_audio), 1):
        try:
            info = sf.info(audio)
        except Exception as error:  # report all unreadable headers together
            failures.append(f"{audio}: {error}")
            continue
        sample_rates[info.samplerate] += 1
        channel_counts[info.channels] += 1
        formats[info.format] += 1
        if info.frames <= 0:
            failures.append(f"{audio}: zero audio frames")
        if index % 1000 == 0:
            print(f"Verified audio headers: {index}/{len(unique_audio)}", flush=True)

    if failures:
        preview = "\n".join(failures[:20])
        raise RuntimeError(
            f"Audio verification failed for {len(failures)} files. First failures:\n{preview}"
        )
    if set(sample_rates) != {16000}:
        raise RuntimeError(f"Expected only 16 kHz audio, found: {dict(sample_rates)}")

    return {
        "unique_audio_files": len(unique_audio),
        "sample_rates": {str(key): value for key, value in sorted(sample_rates.items())},
        "channels": {str(key): value for key, value in sorted(channel_counts.items())},
        "formats": dict(sorted(formats.items())),
    }


def build_readme(
    database: Path,
    ramc: Path,
    notsofar: Path,
    summaries: dict[str, Any],
) -> str:
    lines = [
        "# Consolidated DiariZen diarization data",
        "",
        "This directory is generated by `recipes/diar_ssl/prepare_diarizen_all.py`.",
        "It contains text manifests only; audio remains in its source dataset location and",
        "is referenced by absolute path from each `wav.scp`.",
        "",
        "## Ready-to-use combined splits",
        "",
        "- `train/{wav.scp,rttm,all.uem}`",
        "- `dev/{wav.scp,rttm,all.uem}`",
        "- `test/{wav.scp,rttm,all.uem}`",
        "",
        "Combined recording IDs use `<dataset>__<original_recording_id>` to prevent",
        "cross-dataset ID collisions. Per-dataset files under `by_dataset/` preserve the",
        "original recording IDs.",
        "",
        "Point a DiariZen recipe at the combined files, for example:",
        "",
        "```toml",
        "[train_dataset.args]",
        'scp_file = "data/DiariZen_all/train/wav.scp"',
        'rttm_file = "data/DiariZen_all/train/rttm"',
        'uem_file = "data/DiariZen_all/train/all.uem"',
        "",
        "[validate_dataset.args]",
        'scp_file = "data/DiariZen_all/dev/wav.scp"',
        'rttm_file = "data/DiariZen_all/dev/rttm"',
        'uem_file = "data/DiariZen_all/dev/all.uem"',
        "```",
        "",
        "## Inputs",
        "",
        f"- pyannote manifest: `{database}`",
        f"- RAMC processed data: `{ramc}`",
        f"- NOTSOFAR processed data: `{notsofar}`",
        "",
        "## Recording counts",
        "",
        "| Dataset | Train | Dev | Test |",
        "|---|---:|---:|---:|",
    ]
    per_dataset = summaries["by_dataset"]
    for dataset in per_dataset:
        lines.append(
            f"| {dataset} | "
            f"{per_dataset[dataset].get('train', {}).get('recordings', 0)} | "
            f"{per_dataset[dataset].get('dev', {}).get('recordings', 0)} | "
            f"{per_dataset[dataset].get('test', {}).get('recordings', 0)} |"
        )
    lines.extend(
        [
            f"| **Combined** | {summaries['combined']['train']['recordings']} | "
            f"{summaries['combined']['dev']['recordings']} | "
            f"{summaries['combined']['test']['recordings']} |",
            "",
            "`manifest.json` contains segment counts, annotated hours, source paths, and",
            "audio-header verification statistics.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--ramc", type=Path, default=DEFAULT_RAMC)
    parser.add_argument("--notsofar", type=Path, default=DEFAULT_NOTSOFAR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verify-audio",
        action="store_true",
        help="Read every audio header and require all recordings to be non-empty 16 kHz audio.",
    )
    args = parser.parse_args()

    database = args.database.resolve()
    ramc = args.ramc.resolve()
    notsofar = args.notsofar.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Remove or move it before regenerating."
        )

    manifest_data = yaml.safe_load(database.read_text(encoding="utf-8"))
    splits = discover_pyannote_splits(database, manifest_data)
    for dataset, root in (("RAMC", ramc), ("NOTSOFAR", notsofar)):
        for split_name in ("train", "dev", "test"):
            source_dir = root / split_name
            if source_dir.is_dir():
                splits.append(prepare_existing_split(dataset, root, split_name))

    pair_counts = Counter((item.dataset, item.split) for item in splits)
    duplicate_pairs = sorted(pair for pair, count in pair_counts.items() if count > 1)
    if duplicate_pairs:
        raise ValueError(f"Duplicate dataset/split definitions: {duplicate_pairs}")

    combined = [combine_splits(splits, name) for name in ("train", "dev", "test")]
    audio_summary = verify_audio(splits) if args.verify_audio else {"verified": False}

    summaries: dict[str, Any] = {"by_dataset": {}, "combined": {}}
    for item in splits:
        summaries["by_dataset"].setdefault(item.dataset, {})[item.split] = item.summary()
    for item in combined:
        summaries["combined"][item.split] = item.summary()

    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")

    try:
        staging.mkdir(parents=True)
        for item in splits:
            write_split(staging / "by_dataset" / item.dataset / item.split, item)
        for item in combined:
            write_split(staging / item.split, item)

        metadata = {
            "format": "DiariZen Kaldi-style wav.scp + RTTM + UEM",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "database_yml": str(database),
            "additional_processed_datasets": {
                "RAMC": str(ramc),
                "NOTSOFAR": str(notsofar),
            },
            "recording_id_namespace": "<dataset>__<original_recording_id>",
            "audio_verification": audio_summary,
            **summaries,
        }
        write_lines(
            staging / "manifest.json",
            [json.dumps(metadata, indent=2, sort_keys=True)],
        )
        write_lines(
            staging / "README.md",
            [build_readme(database, ramc, notsofar, summaries).rstrip()],
        )
        os.replace(staging, output)
    except Exception:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise

    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"Prepared consolidated data at {output}")


if __name__ == "__main__":
    main()
