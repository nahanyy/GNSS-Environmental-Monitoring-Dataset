"""
This script reads hourly observation and satelliteInformation JSON files, aligns them to a reference satellite visibility cycle, and generates satellite-time feature matrices.
"""

import json
import gzip
import math
from pathlib import Path
from collections import Counter

import numpy as np


# =========================================================
# Configuration
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent

# Input folder produced by the first preprocessing script.
HOURLY_ROOT = PROJECT_ROOT / "Processed_Data"

# Output folder for aligned satellite-time feature matrices.
OUTPUT_ROOT = PROJECT_ROOT / "Feature_Matrices"

SITE_ID = "s9"

# Source folders for the site to be aligned.
SOURCE_DIRS = [
    HOURLY_ROOT / "s9" / "20260516",
    HOURLY_ROOT / "s9" / "20260517",
]

# Reference folder covering a complete 24-hour satellite visibility cycle.
REFERENCE_DIR = HOURLY_ROOT / "s2" / "20260507"

TARGET_SECONDS = 24 * 60 * 60

ALIGN_CONSTELLATION = "GPS"
MAX_OFFSET_SEC = 1800
COARSE_STEP_SEC = 30
FINE_RANGE_SEC = 90
SAMPLE_STEP_SEC = 60

SAVE_GZIP = False
SAVE_WITH_METADATA = True


# =========================================================
# GNSS configuration
# =========================================================
CONSTELLATIONS = {
    "GPS": {
        "sat_suffix": "G",
        "n_sat": 32,
        "freqs": {
            "L1": "G1",
            "L2": "G2",
        },
        "row_definition": "Rows 1-32 correspond to GPS satellite identifiers PRN 1-32.",
    },
    "Galileo": {
        "sat_suffix": "E",
        "n_sat": 36,
        "freqs": {
            "E1": "E1",
            "E5b": "E2",
        },
        "row_definition": "Rows 1-36 correspond to Galileo satellite identifiers 1-36.",
    },
    "BeiDou": {
        "sat_suffix": "B",
        "n_sat": 64,
        "freqs": {
            "B1": "B1",
            "B2": "B2",
        },
        "row_definition": (
            "Rows 1-64 are reserved for BeiDou satellite identifiers. "
            "Valid svId values mainly cover 1-63, and row 64 may be a placeholder."
        ),
    },
    "QZSS": {
        "sat_suffix": "Q",
        "n_sat": 10,
        "freqs": {
            "L1": "Q1",
            "L2": "Q2",
        },
        "row_definition": "Rows 1-10 correspond to QZSS satellite identifiers 1-10.",
    },
    "GLONASS": {
        "sat_suffix": "R",
        "n_sat": 32,
        "freqs": {
            "L1": "R1",
            "L2": "R2",
        },
        "row_definition": "Rows 1-32 correspond to GLONASS satellite identifiers 1-32.",
    },
}

FEATURES = [
    "cno",
    "prMes",
    "prRes",
    "doMes",
    "elev",
    "azim",
    "cpMes",
    "quality",
    "svUsed",
]


ARRAY_CACHE = {}


# =========================================================
# Basic utilities
# =========================================================
def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if SAVE_GZIP:
        gz_path = Path(str(path) + ".gz")
        with gzip.open(gz_path, "wt", encoding="utf-8") as file:
            json.dump(
                obj,
                file,
                ensure_ascii=False,
                allow_nan=True,
                separators=(",", ":"),
            )
        print(f"Saved: {gz_path}")
    else:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                obj,
                file,
                ensure_ascii=False,
                allow_nan=True,
                separators=(",", ":"),
            )
        print(f"Saved: {path}")


def to_float(value):
    if value is None:
        return np.nan

    if isinstance(value, str):
        value = value.strip()

        if value == "" or value.lower() in {"nan", "none", "null"}:
            return np.nan

        return float(value)

    return float(value)


def to_float_array(value):
    try:
        return np.asarray(value, dtype=float)
    except Exception:
        array = np.asarray(value, dtype=object)
        output = np.empty(array.shape, dtype=float)

        for index in np.ndindex(array.shape):
            output[index] = to_float(array[index])

        return output


def get_cached_array(data_dict, key):
    cache_key = (id(data_dict), key)

    if cache_key not in ARRAY_CACHE:
        ARRAY_CACHE[cache_key] = to_float_array(data_dict[key])

    return ARRAY_CACHE[cache_key]


def record_time_key(data):
    for key in ["recordTime", "recordtime", "time", "Time"]:
        if key in data:
            return key

    raise KeyError("No recordTime key found in this JSON file.")


def record_to_second_of_day(record_time):
    if isinstance(record_time, (int, float)):
        return int(round(record_time)) % TARGET_SECONDS

    text = str(record_time).strip()
    text = text.replace("T", " ")
    text = text.replace("/", "-")

    if "." in text:
        text = text.split(".")[0]

    if len(text) >= 19 and text[13] == "-" and text[16] == "-":
        text = text[:13] + ":" + text[14:16] + ":" + text[17:]

    if " " in text:
        time_part = text.split(" ")[-1]
    else:
        time_part = text

    hour, minute, second = time_part.split(":")
    return int(hour) * 3600 + int(minute) * 60 + int(second)


def normalize_vector(vector, n_sat):
    vector = np.asarray(vector, dtype=float).reshape(-1)

    if len(vector) == n_sat + 1:
        vector = vector[1:]

    output = np.full(n_sat, np.nan, dtype=float)
    length = min(n_sat, len(vector))
    output[:length] = vector[:length]

    return output


def get_vector(entry, candidate_keys, n_sat):
    if entry is None:
        return None

    data = entry["data"]
    index = entry["idx"]
    n_time = entry["n_time"]

    key = None

    for candidate in candidate_keys:
        if candidate in data:
            key = candidate
            break

    if key is None:
        return None

    array = get_cached_array(data, key)

    if array.ndim == 1:
        if len(array) >= n_sat:
            return normalize_vector(array, n_sat)
        return None

    if array.ndim == 2:
        if array.shape[0] == n_time and index < array.shape[0]:
            return normalize_vector(array[index, :], n_sat)

        if array.shape[1] == n_time and index < array.shape[1]:
            return normalize_vector(array[:, index], n_sat)

        if index < array.shape[0]:
            return normalize_vector(array[index, :], n_sat)

    return None


# =========================================================
# Field name mapping
# =========================================================
def key_variants(base, suffix):
    return [
        f"{base}_{suffix}",
        f"{base}{suffix}",
        f"{base.lower()}_{suffix}",
        f"{base.upper()}_{suffix}",
        f"{base.capitalize()}_{suffix}",
    ]


def satellite_key_candidates(feature, sat_suffix):
    if feature == "elev":
        return key_variants("elev", sat_suffix) + key_variants("Elev", sat_suffix)

    if feature == "azim":
        return key_variants("azim", sat_suffix) + key_variants("Azim", sat_suffix)

    if feature == "prRes":
        return key_variants("prRes", sat_suffix)

    if feature == "quality":
        return (
            key_variants("qualityInd", sat_suffix)
            + key_variants("Quality", sat_suffix)
            + key_variants("quality", sat_suffix)
        )

    if feature == "svUsed":
        return key_variants("svUsed", sat_suffix)

    if feature == "cno":
        return key_variants("cno", sat_suffix) + key_variants("cn0", sat_suffix)

    return []


def observation_key_candidates(feature, obs_suffix):
    if feature == "cno":
        return key_variants("cn0", obs_suffix) + key_variants("cno", obs_suffix)

    if feature in {"prMes", "doMes", "cpMes"}:
        return key_variants(feature, obs_suffix)

    return []


def feature_sources(feature, sat_suffix, obs_suffix):
    if feature == "cno":
        return [
            ("obs", observation_key_candidates("cno", obs_suffix)),
            ("sat", satellite_key_candidates("cno", sat_suffix)),
        ]

    if feature in {"prMes", "doMes", "cpMes"}:
        return [
            ("obs", observation_key_candidates(feature, obs_suffix)),
        ]

    if feature in {"prRes", "elev", "azim", "quality", "svUsed"}:
        return [
            ("sat", satellite_key_candidates(feature, sat_suffix)),
        ]

    return []


# =========================================================
# Hourly file loading
# =========================================================
def find_hourly_json_files(folder: Path, kind: str):
    files = []

    for path in folder.rglob("*.json"):
        name = path.name.lower()

        if kind == "obs":
            if name.startswith("observation") and "pvt" not in name:
                files.append(path)

        elif kind == "sat":
            if "satelliteinformation" in name or "satelliteinfomation" in name:
                files.append(path)

    return sorted(files)


def load_hourly_folder(folder: Path):
    print(f"\nLoading folder: {folder}")

    result = {
        "dir": project_relative(folder),
        "obs": {},
        "sat": {},
    }

    for kind in ["obs", "sat"]:
        files = find_hourly_json_files(folder, kind)
        print(f"  {kind}: {len(files)} files")

        for path in files:
            try:
                data = load_json(path)
                time_key = record_time_key(data)
                record_times = data[time_key]
            except Exception as error:
                print(f"  Warning: failed to read {path}: {error}")
                continue

            if not isinstance(record_times, list):
                record_times = [record_times]

            n_time = len(record_times)

            for index, record_time in enumerate(record_times):
                try:
                    second = record_to_second_of_day(record_time)
                except Exception:
                    continue

                if 0 <= second < TARGET_SECONDS:
                    result[kind][second] = {
                        "data": data,
                        "idx": index,
                        "n_time": n_time,
                        "file": project_relative(path),
                    }

    print(f"  Loaded observation epochs: {len(result['obs'])}")
    print(f"  Loaded satelliteInformation epochs: {len(result['sat'])}")

    return result


def has_epoch(day_data, second):
    return second in day_data["obs"] or second in day_data["sat"]


# =========================================================
# Satellite-position alignment
# =========================================================
def circular_azimuth_diff(a, b):
    diff = np.abs(a - b)
    return np.minimum(diff, 360.0 - diff)


def get_angle_vectors(day_data, second, constellation_name=ALIGN_CONSTELLATION):
    cfg = CONSTELLATIONS[constellation_name]
    sat_suffix = cfg["sat_suffix"]
    n_sat = cfg["n_sat"]

    entry = day_data["sat"].get(second)

    if entry is None:
        return None, None

    elev = get_vector(entry, satellite_key_candidates("elev", sat_suffix), n_sat)
    azim = get_vector(entry, satellite_key_candidates("azim", sat_suffix), n_sat)

    return elev, azim


def angle_score(ref_data, src_data, ref_sec, src_sec, constellation_name=ALIGN_CONSTELLATION):
    ref_elev, ref_azim = get_angle_vectors(ref_data, ref_sec, constellation_name)
    src_elev, src_azim = get_angle_vectors(src_data, src_sec, constellation_name)

    if ref_elev is None or ref_azim is None or src_elev is None or src_azim is None:
        return math.inf

    valid_ref = np.isfinite(ref_elev) & np.isfinite(ref_azim)
    valid_src = np.isfinite(src_elev) & np.isfinite(src_azim)

    valid_ref &= (np.abs(ref_elev) > 1e-9) | (np.abs(ref_azim) > 1e-9)
    valid_src &= (np.abs(src_elev) > 1e-9) | (np.abs(src_azim) > 1e-9)

    valid = valid_ref & valid_src

    if np.sum(valid) < 4:
        return math.inf

    elevation_diff = np.abs(ref_elev[valid] - src_elev[valid])
    azimuth_diff = circular_azimuth_diff(ref_azim[valid], src_azim[valid])

    return float(np.mean(elevation_diff + azimuth_diff))


def mean_score_for_offset(ref_data, src_data, offset_sec):
    scores = []

    for ref_sec in range(0, TARGET_SECONDS, SAMPLE_STEP_SEC):
        src_sec = ref_sec + offset_sec

        if src_sec < 0 or src_sec >= TARGET_SECONDS:
            continue

        if not has_epoch(src_data, src_sec):
            continue

        score = angle_score(ref_data, src_data, ref_sec, src_sec)

        if np.isfinite(score):
            scores.append(score)

    if len(scores) < 20:
        return math.inf

    return float(np.mean(scores))


def estimate_offset(ref_data, src_data):
    print(f"\nEstimating offset for source: {src_data['dir']}")

    best_offset = None
    best_score = math.inf

    for offset in range(-MAX_OFFSET_SEC, MAX_OFFSET_SEC + 1, COARSE_STEP_SEC):
        score = mean_score_for_offset(ref_data, src_data, offset)

        if score < best_score:
            best_score = score
            best_offset = offset

    if best_offset is None:
        print("  Warning: failed to estimate offset. Offset is set to 0.")
        return 0, math.inf

    fine_start = best_offset - FINE_RANGE_SEC
    fine_end = best_offset + FINE_RANGE_SEC

    for offset in range(fine_start, fine_end + 1):
        score = mean_score_for_offset(ref_data, src_data, offset)

        if score < best_score:
            best_score = score
            best_offset = offset

    print(f"  Best offset: {best_offset} s, mean angle score: {best_score:.4f}")
    return int(best_offset), float(best_score)


def build_epoch_mapping(ref_data, source_list, offsets):
    selected = [None] * TARGET_SECONDS
    source_counter = Counter()

    print("\nBuilding aligned 24-hour epoch mapping...")

    for ref_sec in range(TARGET_SECONDS):
        best = None
        fallback = None
        best_score = math.inf

        for src_index, src_data in enumerate(source_list):
            src_sec = ref_sec + offsets[src_index]

            if src_sec < 0 or src_sec >= TARGET_SECONDS:
                continue

            if not has_epoch(src_data, src_sec):
                continue

            if fallback is None:
                fallback = (src_index, src_sec)

            score = angle_score(ref_data, src_data, ref_sec, src_sec)

            if np.isfinite(score) and score < best_score:
                best_score = score
                best = (src_index, src_sec)

        if best is None:
            best = fallback

        selected[ref_sec] = best

        if best is not None:
            source_counter[source_list[best[0]]["dir"]] += 1

    missing = sum(item is None for item in selected)

    print("Epoch mapping summary:")
    for source_dir, count in source_counter.items():
        print(f"  {source_dir}: {count} epochs")

    print(f"  Missing epochs in output: {missing}")

    return selected, source_counter


# =========================================================
# Feature matrix generation
# =========================================================
def extract_feature_vector(day_data, second, feature, cfg, freq_label):
    sat_suffix = cfg["sat_suffix"]
    obs_suffix = cfg["freqs"][freq_label]
    n_sat = cfg["n_sat"]

    for kind, candidates in feature_sources(feature, sat_suffix, obs_suffix):
        entry = day_data[kind].get(second)
        vector = get_vector(entry, candidates, n_sat)

        if vector is not None:
            return vector

    return None


def generate_feature_matrix(const_name, freq_label, feature, source_list, selected_mapping):
    cfg = CONSTELLATIONS[const_name]
    n_sat = cfg["n_sat"]

    matrix = np.full((n_sat, TARGET_SECONDS), np.nan, dtype=float)

    for ref_sec, selected in enumerate(selected_mapping):
        if selected is None:
            continue

        src_index, src_sec = selected
        src_data = source_list[src_index]

        vector = extract_feature_vector(src_data, src_sec, feature, cfg, freq_label)

        if vector is not None:
            matrix[:, ref_sec] = vector

    return matrix


def matrix_statistics(matrix):
    total_count = int(matrix.size)
    finite_mask = np.isfinite(matrix)
    finite_values = matrix[finite_mask]

    stats = {
        "total_count": total_count,
        "finite_count": int(finite_mask.sum()),
        "nan_count": int(np.isnan(matrix).sum()),
        "zero_count": int(np.sum(matrix == 0)),
        "positive_count": int(np.sum(matrix > 0)),
    }

    if total_count > 0:
        stats["finite_ratio"] = stats["finite_count"] / total_count
        stats["nan_ratio"] = stats["nan_count"] / total_count
        stats["zero_ratio"] = stats["zero_count"] / total_count
        stats["positive_ratio"] = stats["positive_count"] / total_count

    if finite_values.size > 0:
        stats["min"] = float(np.min(finite_values))
        stats["max"] = float(np.max(finite_values))
        stats["mean"] = float(np.mean(finite_values))
        stats["std"] = float(np.std(finite_values))
    else:
        stats["min"] = None
        stats["max"] = None
        stats["mean"] = None
        stats["std"] = None

    return stats


def save_feature_matrix(site_id, const_name, freq_label, feature, matrix, offsets, source_counter):
    out_dir = OUTPUT_ROOT / site_id / const_name / freq_label / feature
    out_file = out_dir / f"{site_id}_{const_name}_{freq_label}_{feature}.json"

    cfg = CONSTELLATIONS[const_name]

    if SAVE_WITH_METADATA:
        obj = {
            "site": site_id,
            "constellation": const_name,
            "frequency": freq_label,
            "feature": feature,
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "row_definition": cfg["row_definition"],
            "column_definition": (
                "Columns correspond to aligned seconds of a complete 24-hour reference cycle, "
                "from 00:00:00 to 23:59:59."
            ),
            "missing_value": "NaN indicates missing, invalid, or unavailable observations after alignment.",
            "zero_value_note": (
                "Zero values are retained from the receiver output and should not be directly "
                "treated as missing values."
            ),
            "alignment": {
                "reference_dir": project_relative(REFERENCE_DIR),
                "source_dirs": [project_relative(path) for path in SOURCE_DIRS],
                "mapping_rule": "source_second = reference_second + offset_second",
                "offset_seconds": offsets,
                "source_epoch_counts": dict(source_counter),
                "alignment_constellation": ALIGN_CONSTELLATION,
            },
            "statistics": matrix_statistics(matrix),
            "data": matrix.tolist(),
        }
    else:
        obj = matrix.tolist()

    save_json(obj, out_file)


def save_alignment_log(offsets, offset_scores, source_counter):
    log_file = OUTPUT_ROOT / SITE_ID / f"{SITE_ID}_alignment_log.json"

    obj = {
        "site": SITE_ID,
        "processing_type": "satellite-position-based cross-cycle alignment",
        "reference_dir": project_relative(REFERENCE_DIR),
        "source_dirs": [project_relative(path) for path in SOURCE_DIRS],
        "mapping_rule": "source_second = reference_second + offset_second",
        "offset_seconds": offsets,
        "offset_scores": offset_scores,
        "source_epoch_counts": dict(source_counter),
        "target_seconds": TARGET_SECONDS,
        "alignment_constellation": ALIGN_CONSTELLATION,
        "notes": [
            "The input files are hourly observation and satelliteInformation JSON files generated by the raw UBX preprocessing script.",
            "NaN is used for missing, invalid, or unavailable observations.",
            "Zero values are retained from the receiver output.",
            "For BeiDou, row 64 may be a placeholder because valid svId values mainly cover 1-63.",
            "Some receiver outputs may use -91 to indicate invalid elevation angles.",
        ],
    }

    save_json(obj, log_file)


# =========================================================
# Main workflow
# =========================================================
def main():
    print("Step 1: Load reference data")
    ref_data = load_hourly_folder(REFERENCE_DIR)

    print("\nStep 2: Load source data")
    source_list = [load_hourly_folder(path) for path in SOURCE_DIRS]

    print("\nStep 3: Estimate satellite-position offsets")
    offsets = []
    offset_scores = []

    for source_data in source_list:
        offset, score = estimate_offset(ref_data, source_data)
        offsets.append(offset)
        offset_scores.append(score if np.isfinite(score) else None)

    print("\nFinal offsets:")
    for source_dir, offset, score in zip(SOURCE_DIRS, offsets, offset_scores):
        print(f"  {project_relative(source_dir)}: offset = {offset} s, score = {score}")

    print("\nStep 4: Build aligned epoch mapping")
    selected_mapping, source_counter = build_epoch_mapping(ref_data, source_list, offsets)

    save_alignment_log(
        offsets=offsets,
        offset_scores=offset_scores,
        source_counter=source_counter,
    )

    print("\nStep 5: Generate satellite-time feature matrices")
    for const_name, cfg in CONSTELLATIONS.items():
        for freq_label in cfg["freqs"]:
            for feature in FEATURES:
                print(f"  Generating: {SITE_ID} / {const_name} / {freq_label} / {feature}")

                matrix = generate_feature_matrix(
                    const_name=const_name,
                    freq_label=freq_label,
                    feature=feature,
                    source_list=source_list,
                    selected_mapping=selected_mapping,
                )

                save_feature_matrix(
                    site_id=SITE_ID,
                    const_name=const_name,
                    freq_label=freq_label,
                    feature=feature,
                    matrix=matrix,
                    offsets=offsets,
                    source_counter=source_counter,
                )

    print("\nDone.")


if __name__ == "__main__":
    main()