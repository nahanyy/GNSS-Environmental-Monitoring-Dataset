"""
Satellite-geometry-based cross-site alignment and satellite-time matrix generation.

This script aligns GNSS observations from different monitoring sites using
satellite geometry information (elevation and azimuth angle). The observations
from the reference site are used as the alignment baseline, and the source
sites are temporally aligned to the reference site to generate satellite-time
feature matrices.
"""
import json
import math
import gzip
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import Counter

SITE_ID = "S1" # Name of the acquisition environment that needs to be aligned

REFERENCE_DIR = Path(
    r"..\Processed data\5_7"
)  # The alignment reference data path originates from acquisition point S2

SOURCE_DIRS = [
    Path(
        r"..\Processed data\5_5"
    ),
    Path(
        r"..\Processed data\5_6"
    )
]  # Path to the data collection points (e.g., S1) that need to be aligned

OUTPUT_ROOT = Path(
    r"E:\aligned_result1"
)

TARGET_SECONDS = 86400
ALIGN_CONSTELLATION = "GPS"
MAX_OFFSET_SEC = TARGET_SECONDS // 2
COARSE_STEP_SEC = 300
FINE_RANGE_SEC = COARSE_STEP_SEC
FINE_STEP_SEC = 10
MIN_ALIGN_ELEV_DEG = 10
MIN_VALID_ALIGN_EPOCHS = 1000
SAMPLE_STEP_SEC = 30
SAVE_GZIP = False
SAVE_WITH_METADATA = True

CONSTELLATIONS = {
    "GPS": {
        "sat_suffix": "G",
        "n_sat": 32,
        "freqs": {
            "L1": "G1",
            "L2": "G2"
        },
        "row_definition":
            "Rows 1-32 correspond to GPS satellite identifiers PRN 1-32."
    },

    "Galileo": {
        "sat_suffix": "E",
        "n_sat": 36,
        "freqs": {
            "E1": "E1",
            "E5b": "E2"
        },
        "row_definition":
            "Rows 1-36 correspond to Galileo satellite identifiers 1-36."
    },

    "BeiDou": {
        "sat_suffix": "B",
        "n_sat": 64,
        "freqs": {
            "B1": "B1",
            "B2": "B2"
        },
        "row_definition":
            "Rows 1-64 correspond to BeiDou satellite identifiers."
    },

    "QZSS": {
        "sat_suffix": "Q",
        "n_sat": 10,
        "freqs": {
            "L1": "Q1",
            "L2": "Q2"
        },
        "row_definition":
            "Rows 1-10 correspond to QZSS satellite identifiers."
    },

    "GLONASS": {
        "sat_suffix": "R",
        "n_sat": 32,
        "freqs": {
            "L1": "R1",
            "L2": "R2"
        },
        "row_definition":
            "Rows 1-32 correspond to GLONASS satellite identifiers."
    }
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
    "svUsed"
]

ARRAY_CACHE = {}
def project_relative(path):
    try:
        return str(
            path.resolve()
        )
    except:
        return str(path)

def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)

def save_json(obj,path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    if SAVE_GZIP:
        gz_path = Path(
            str(path)+".gz"
        )
        with gzip.open(
            gz_path,
            "wt",
            encoding="utf-8"
        ) as f:
            json.dump(
                obj,
                f,
                ensure_ascii=False,
                allow_nan=True
            )
        print(
            "Saved:",
            gz_path
        )

    else:
        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                obj,
                f,
                ensure_ascii=False,
                allow_nan=True
            )
        print(
            "Saved:",
            path
        )

def to_float_array(value):
    return np.asarray(
        value,
        dtype=float
    )

def get_cached_array(data,key):
    cache_key=(id(data),key)
    if cache_key not in ARRAY_CACHE:
        ARRAY_CACHE[cache_key]=to_float_array(
            data[key]
        )
    return ARRAY_CACHE[cache_key]

def find_json_files(folder,kind):
    files=[]
    for p in folder.rglob("*.json"):
        name=p.name.lower()
        if kind=="obs":
            if (
                name.startswith("observation")
                and
                "pvt" not in name
            ):
                files.append(p)

        elif kind=="sat":
            if (
                "satelliteinformation" in name
                or
                "satelliteinfomation" in name
            ):
                files.append(p)

    return sorted(files)

def record_time_key(data):
    for k in [
        "recordTime",
        "recordtime",
        "time",
        "Time"
    ]:
        if k in data:
            return k

    raise KeyError(
        "No record time"
    )

def record_to_second(record):
    text=str(record)
    text=text.replace(
        "T",
        " "
    )

    if " " in text:
        text=text.split(" ")[1]

    if "." in text:
        text=text.split(".")[0]
    h,m,s=text.split(":")

    return (
        int(h)*3600
        +
        int(m)*60
        +
        int(s)
    )

def load_hourly_folder(folder):
    print(
        "\nLoading:",
        folder
    )
    result={
        "dir":
            project_relative(folder),
        "obs":{},
        "sat":{}
    }

    for kind in [
        "obs",
        "sat"
    ]:
        files=find_json_files(
            folder,
            kind
        )
        print(
            kind,
            len(files),
            "files"
        )

        for path in files:
            data=load_json(path)
            key=record_time_key(data)
            times=data[key]
            n_time=len(times)

            for idx,t in enumerate(times):
                sec=record_to_second(t)
                result[kind][sec]={
                    "data":data,
                    "idx":idx,
                    "n_time":n_time
                }
    print(
        "obs epochs:",
        len(result["obs"])
    )
    print(
        "sat epochs:",
        len(result["sat"])
    )
    return result

def normalize_vector(vector, n_sat):
    vector=np.asarray(
        vector,
        dtype=float
    ).reshape(-1)

    output=np.full(
        n_sat,
        np.nan,
        dtype=float
    )

    length=min(
        len(vector),
        n_sat
    )

    output[:length]=vector[:length]

    return output


def key_variants(base,suffix):
    return [
        f"{base}_{suffix}",
        f"{base}{suffix}",
        f"{base.lower()}_{suffix}",
        f"{base.upper()}_{suffix}"
    ]

def satellite_key_candidates(
        feature,
        suffix):
    if feature=="elev":
        return key_variants(
            "elev",
            suffix
        )
    if feature=="azim":
        return key_variants(
            "azim",
            suffix
        )
    if feature=="svUsed":
        return key_variants(
            "svUsed",
            suffix
        )
    if feature=="svId":
        return key_variants(
            "svId",
            suffix
        )
    if feature=="quality":
        return (
            key_variants(
                "qualityInd",
                suffix
            )
            +
            key_variants(
                "quality",
                suffix
            )
        )

    return []


def observation_key_candidates(
        feature,
        suffix):
    if feature=="cno":
        return (
            key_variants(
                "cno",
                suffix
            )
            +
            key_variants(
                "cn0",
                suffix
            )
        )

    if feature in [
        "prMes",
        "doMes",
        "cpMes"
    ]:
        return key_variants(
            feature,
            suffix
        )
    return []

def get_vector(
        entry,
        candidates,
        n_sat):
    if entry is None:
        return None
    data=entry["data"]
    idx=entry["idx"]
    n_time=entry["n_time"]
    key=None
    for c in candidates:
        if c in data:
            key=c
            break

    if key is None:
        return None

    array=get_cached_array(
        data,
        key
    )
    if array.ndim==1:
        return normalize_vector(
            array,
            n_sat
        )
    if array.ndim==2:
        if array.shape[0]==n_time:
            return normalize_vector(
                array[idx,:],
                n_sat
            )
        if array.shape[1]==n_time:
            return normalize_vector(
                array[:,idx],
                n_sat
            )
    return None


def build_alignment_satellite_cache(day_data):
    """
    Build the GPS satellite dictionary used only for temporal alignment.

    The logic follows the second alignment script:
    each epoch is represented as {PRN: {"elev": ..., "azim": ...}},
    so reference and source observations are compared by the same svId_G
    rather than by array position.
    """
    if "align_sat" in day_data:
        return day_data["align_sat"]

    cfg=CONSTELLATIONS[
        ALIGN_CONSTELLATION
    ]
    suffix=cfg["sat_suffix"]
    n_sat=cfg["n_sat"]
    cache={}

    for second,entry in day_data["sat"].items():
        elev=get_vector(
            entry,
            satellite_key_candidates(
                "elev",
                suffix
            ),
            n_sat
        )
        azim=get_vector(
            entry,
            satellite_key_candidates(
                "azim",
                suffix
            ),
            n_sat
        )
        svid=get_vector(
            entry,
            satellite_key_candidates(
                "svId",
                suffix
            ),
            n_sat
        )

        if (
            elev is None
            or
            azim is None
            or
            svid is None
        ):
            continue

        sat={}
        for j in range(n_sat):
            if not np.isfinite(svid[j]):
                continue

            prn=int(svid[j])
            if prn<=0:
                continue

            if (
                not np.isfinite(elev[j])
                or
                not np.isfinite(azim[j])
            ):
                continue

            sat[prn]={
                "elev":float(elev[j]),
                "azim":float(azim[j])
            }

        cache[second]=sat

    day_data["align_sat"]=cache
    return cache


def get_alignment_satellites(
        day_data,
        second):
    cache=build_alignment_satellite_cache(
        day_data
    )
    return cache.get(second)


def circular_azimuth_diff(a,b):
    diff=np.abs(
        a-b
    )
    return np.minimum(
        diff,
        360-diff
    )


def epoch_alignment_error(
        ref_data,
        src_data,
        ref_sec,
        src_sec):
    """
    Compare one reference/source epoch using the same logic as the
    second script: common GPS PRNs only, with elevation >= 10 degrees
    at both sites.
    """
    ref=get_alignment_satellites(
        ref_data,
        ref_sec
    )
    src=get_alignment_satellites(
        src_data,
        src_sec
    )

    if ref is None or src is None:
        return None

    common=(
        set(ref.keys())
        &
        set(src.keys())
    )

    elev=[]
    azim=[]

    for prn in common:
        r=ref[prn]
        t=src[prn]

        if r["elev"]<MIN_ALIGN_ELEV_DEG:
            continue

        if t["elev"]<MIN_ALIGN_ELEV_DEG:
            continue

        elev.append(
            abs(
                r["elev"]
                -
                t["elev"]
            )
        )
        azim.append(
            float(
                circular_azimuth_diff(
                    r["azim"],
                    t["azim"]
                )
            )
        )

    if len(elev)==0:
        return None

    return (
        np.asarray(elev,dtype=float),
        np.asarray(azim,dtype=float),
        len(elev)
    )


def angle_score(
        ref_data,
        src_data,
        ref_sec,
        src_sec):
    e=epoch_alignment_error(
        ref_data,
        src_data,
        ref_sec,
        src_sec
    )

    if e is None:
        return math.inf

    return float(
        np.mean(e[0])
        +
        np.mean(e[1])
    )


def mean_score_for_offset(
        ref_data,
        src_data,
        offset):
    """
    Score one cycle offset using the same pooled-satellite criterion as
    calculate_score() in the second script:

        score = mean(all elevation errors) + mean(all azimuth errors)

    offset keeps the original first-script convention:
        src_second = (ref_second + offset) mod 86400
    """
    elev=[]
    azim=[]
    valid_epochs=0

    src_cache=build_alignment_satellite_cache(
        src_data
    )
    build_alignment_satellite_cache(
        ref_data
    )

    for src_sec in src_cache.keys():
        ref_sec=(
            src_sec-offset
        ) % TARGET_SECONDS

        e=epoch_alignment_error(
            ref_data,
            src_data,
            ref_sec,
            src_sec
        )

        if e is None:
            continue

        elev.extend(e[0])
        azim.extend(e[1])
        valid_epochs+=1

    if valid_epochs<MIN_VALID_ALIGN_EPOCHS:
        return math.inf

    if len(elev)==0:
        return math.inf

    return float(
        np.mean(elev)
        +
        np.mean(azim)
    )


def estimate_offset(
        ref_data,
        src_data):

    print(
        "\nEstimate offset:",
        src_data["dir"]
    )

    build_alignment_satellite_cache(
        ref_data
    )
    build_alignment_satellite_cache(
        src_data
    )

    best_offset=0
    best_score=math.inf

    # Full 24-hour cycle coarse search. As in the second script, the
    # candidate is the start position of the source segment in the
    # reference cycle. The corresponding value is converted to the
    # original first-script offset convention.
    src_seconds=sorted(
        src_data["align_sat"].keys()
    )
    if len(src_seconds)==0:
        raise RuntimeError(
            "No valid GPS satellite epochs were found for alignment."
        )

    source_start_sec=src_seconds[0]
    coarse_offsets=[]
    for ref_start_sec in range(
        0,
        TARGET_SECONDS,
        COARSE_STEP_SEC
    ):
        raw_offset=(
            source_start_sec
            -
            ref_start_sec
        )
        offset=(
            (raw_offset+MAX_OFFSET_SEC)
            % TARGET_SECONDS
        )-MAX_OFFSET_SEC
        coarse_offsets.append(offset)

    for offset in tqdm(
        coarse_offsets
    ):
        score=mean_score_for_offset(
            ref_data,
            src_data,
            offset
        )
        if score<best_score:
            best_score=score
            best_offset=offset

    if not np.isfinite(best_score):
        raise RuntimeError(
            "No valid alignment offset was found. "
            "Please check svId_G/elev_G/azim_G and the number of valid epochs."
        )

    # Fine search around the coarse optimum. The second script uses
    # a +/-300 s neighborhood and a 10 s fine step.
    fine_candidates=[]
    for raw_offset in range(
        best_offset-FINE_RANGE_SEC,
        best_offset+FINE_RANGE_SEC+1,
        FINE_STEP_SEC
    ):
        # Normalize to the signed representation of one 24-hour cycle.
        offset=(
            (raw_offset+MAX_OFFSET_SEC)
            % TARGET_SECONDS
        )-MAX_OFFSET_SEC
        fine_candidates.append(offset)

    for offset in dict.fromkeys(fine_candidates):
        score=mean_score_for_offset(
            ref_data,
            src_data,
            offset
        )

        if score<best_score:
            best_score=score
            best_offset=offset

    print(
        "Best offset:",
        best_offset,
        "score:",
        best_score
    )
    return int(best_offset),float(best_score)


def build_epoch_mapping(
        ref_data,
        source_list,
        offsets):
    selected=[None]*TARGET_SECONDS
    counter=Counter()
    print(
        "\nBuild 24h mapping..."
    )

    build_alignment_satellite_cache(
        ref_data
    )
    for src in source_list:
        build_alignment_satellite_cache(
            src
        )

    for ref_sec in range(
        TARGET_SECONDS
    ):
        best=None
        fallback=None
        best_score=math.inf
        for i,src in enumerate(source_list):
            # Keep the original first-script offset interface, but use
            # cyclic 24-hour placement as in insert_cycle() of script 2.
            src_sec=(
                ref_sec+offsets[i]
            ) % TARGET_SECONDS

            if src_sec not in src["sat"]:
                continue

            if fallback is None:
                fallback=(i,src_sec)

            score=angle_score(
                ref_data,
                src,
                ref_sec,
                src_sec
            )

            if score<best_score:
                best_score=score
                best=(i,src_sec)

        if best is None:
            best=fallback

        selected[ref_sec]=best

        if best is not None:
            counter[
                source_list[best[0]]["dir"]
            ]+=1

    print(
        "Mapping finished"
    )
    print(
        "Missing:",
        sum(
            x is None
            for x in selected
        )
    )
    return selected,counter


def extract_feature_vector(
        day_data,
        second,
        feature,
        const_name,
        freq_label):
    cfg=CONSTELLATIONS[const_name]
    sat_suffix=cfg["sat_suffix"]
    n_sat=cfg["n_sat"]
    if feature in [
        "elev",
        "azim",
        "svUsed",
        "quality"
    ]:
        entry=day_data["sat"].get(
            second
        )
        if entry is None:
            return None
        vector=get_vector(
            entry,
            satellite_key_candidates(
                feature,
                sat_suffix
            ),
            n_sat
        )
        return vector

    else:
        obs_suffix=cfg["freqs"][freq_label]
        entry=day_data["obs"].get(
            second
        )
        if entry is None:
            return None

        vector=get_vector(
            entry,
            observation_key_candidates(
                feature,
                obs_suffix
            ),
            n_sat
        )
        return vector


def generate_feature_matrix(
        const_name,
        freq_label,
        feature,
        source_list,
        selected_mapping):
    cfg=CONSTELLATIONS[const_name]
    n_sat=cfg["n_sat"]
    matrix=np.full(
        (
            n_sat,
            TARGET_SECONDS
        ),
        np.nan,
        dtype=float
    )

    for ref_sec,selected in enumerate(
        selected_mapping
    ):
        if selected is None:
            continue
        src_index,src_sec=selected
        src_data=source_list[src_index]
        vector=extract_feature_vector(
            src_data,
            src_sec,
            feature,
            const_name,
            freq_label
        )
        if vector is not None:
            matrix[:,ref_sec]=vector
    return matrix


def calculate_alignment_error(
        ref_data,
        src_data,
        offset,
        name,
        return_errors=False):
    elev_errors=[]
    azim_errors=[]

    build_alignment_satellite_cache(
        ref_data
    )
    src_cache=build_alignment_satellite_cache(
        src_data
    )

    for src_sec in src_cache.keys():
        ref_sec=(
            src_sec-offset
        ) % TARGET_SECONDS

        e=epoch_alignment_error(
            ref_data,
            src_data,
            ref_sec,
            src_sec
        )

        if e is None:
            continue

        elev_errors.extend(e[0])
        azim_errors.extend(e[1])

    elev_errors=np.array(
        elev_errors
    )
    azim_errors=np.array(
        azim_errors
    )

    if (
        len(elev_errors)==0
        or
        len(azim_errors)==0
    ):
        result={
            "name":name,
            "elev_mean":float("nan"),
            "elev_std":float("nan"),
            "elev_95":float("nan"),
            "azim_mean":float("nan"),
            "azim_std":float("nan"),
            "azim_95":float("nan")
        }
        if return_errors:
            return result,elev_errors,azim_errors
        return result

    result={
        "name":name,
        "elev_mean":
            float(
                np.mean(elev_errors)
            ),
        "elev_std":
            float(
                np.std(elev_errors)
            ),
        "elev_95":
            float(
                np.percentile(
                    elev_errors,
                    95
                )
            ),
        "azim_mean":
            float(
                np.mean(azim_errors)
            ),
        "azim_std":
            float(
                np.std(azim_errors)
            ),
        "azim_95":
            float(
                np.percentile(
                    azim_errors,
                    95
                )
            )
    }
    if return_errors:
        return result,elev_errors,azim_errors
    return result


def sensitivity_lines(elev,azim,name):
    lines=[]
    lines.append("========== Sensitivity ==========")
    lines.append(name)

    if (
        elev is None
        or azim is None
        or len(elev)==0
        or len(azim)==0
    ):
        lines.append("No valid aligned samples.")
        return lines

    elev=np.asarray(elev,dtype=float)
    azim=np.asarray(azim,dtype=float)

    for e in [1,2,3,5]:
        for a in [2,5,10,15]:
            rate=np.mean(
                (elev<=e)
                &
                (azim<=a)
            )*100.0

            lines.append(
                f"Elev<={e}°, Azim<={a}° : {rate:.2f}%"
            )

    return lines


def save_alignment_evaluation(
        evaluation_list,
        sensitivity_data=None):

    path=OUTPUT_ROOT / SITE_ID / (
        SITE_ID+
        "_alignment_evaluation.txt"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:
        for item in evaluation_list:
            f.write(
                json.dumps(
                    item,
                    indent=4,
                    ensure_ascii=False
                )
            )
            f.write(
                "\n\n"
            )
    print(
        "Saved:",
        path
    )

    # Additional human-readable alignment report.
    # This does not change the original evaluation file above.
    report_path=OUTPUT_ROOT / SITE_ID / "alignment_report.txt"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:
        for item in evaluation_list:
            name=item["name"]

            f.write("====================\n")
            f.write(f"{name}\n")
            f.write("====================\n")
            f.write(
                f"Elevation Mean: {item['elev_mean']:.6f} deg\n"
            )
            f.write(
                f"Elevation STD: {item['elev_std']:.6f} deg\n"
            )
            f.write(
                f"Elevation 95%: {item['elev_95']:.6f} deg\n"
            )
            f.write(
                f"Azimuth Mean: {item['azim_mean']:.6f} deg\n"
            )
            f.write(
                f"Azimuth STD: {item['azim_std']:.6f} deg\n"
            )
            f.write(
                f"Azimuth 95%: {item['azim_95']:.6f} deg\n\n"
            )

            if (
                sensitivity_data is not None
                and name in sensitivity_data
            ):
                elev,azim=sensitivity_data[name]
                for line in sensitivity_lines(
                    elev,
                    azim,
                    name
                ):
                    f.write(line+"\n")
                f.write("\n")

    print(
        "Saved:",
        report_path
    )


def save_feature_matrix(
        const_name,
        freq_label,
        feature,
        matrix,
        offsets,
        counter):

    out_dir=(
        OUTPUT_ROOT
        /
        SITE_ID
        /
        const_name
        /
        freq_label
        /
        feature
    )

    out_file=out_dir / (
        f"{SITE_ID}_"
        f"{const_name}_"
        f"{freq_label}_"
        f"{feature}.json"
    )

    cfg=CONSTELLATIONS[const_name]

    obj={

        "site":
            SITE_ID,
        "constellation":
            const_name,
        "frequency":
            freq_label,
        "feature":
            feature,
        "shape":
            [
                int(matrix.shape[0]),
                int(matrix.shape[1])
            ],
        "row_definition":
            cfg["row_definition"],
        "column_definition":
            "Columns correspond to aligned seconds of a complete 24-hour reference cycle.",
        "missing_value":
            "NaN indicates missing or unavailable observations after alignment.",
        "alignment":{
            "reference":
                project_relative(
                    REFERENCE_DIR
                ),
            "sources":
                [
                    project_relative(x)
                    for x in SOURCE_DIRS
                ],
            "offset_seconds":
                offsets
        },
        "data":
            matrix.tolist()
    }
    save_json(
        obj,
        out_file
    )

def save_alignment_log(
        offsets,
        scores,
        counter):
    path=OUTPUT_ROOT / SITE_ID / (
        SITE_ID+
        "_alignment_log.json"
    )
    obj={
        "site":
            SITE_ID,
        "method":
            "satellite-position-based cross-cycle alignment",
        "offset_seconds":
            offsets,
        "offset_scores":
            scores,
        "source_epoch_counts":
            dict(counter)
    }
    save_json(
        obj,
        path
    )


def main():
    print(
        "Step 1: Load reference"
    )
    ref_data=load_hourly_folder(
        REFERENCE_DIR
    )
    print(
        "\nStep 2: Load sources"
    )
    source_list=[
        load_hourly_folder(p)
        for p in SOURCE_DIRS
    ]
    print(
        "\nStep 3: Estimate offsets"
    )
    offsets=[]
    offset_scores=[]
    for src in source_list:
        offset,score=estimate_offset(
            ref_data,
            src
        )

        offsets.append(
            offset
        )

        offset_scores.append(
            score
        )

    print(
        "\nStep 4: Build mapping"
    )

    selected_mapping,counter=build_epoch_mapping(
        ref_data,
        source_list,
        offsets
    )

    save_alignment_log(
        offsets,
        offset_scores,
        counter
    )

    print(
        "\nStep 5: Alignment evaluation"
    )
    evaluation=[]
    sensitivity_data={}
    all_elev_errors=[]
    all_azim_errors=[]

    for i,src in enumerate(source_list):
        name=f"Day{i+1}"
        result,elev_errors,azim_errors=(
            calculate_alignment_error(
                ref_data,
                src,
                offsets[i],
                name,
                return_errors=True
            )
        )

        evaluation.append(result)
        sensitivity_data[name]=(
            elev_errors,
            azim_errors
        )
        all_elev_errors.extend(elev_errors)
        all_azim_errors.extend(azim_errors)

    # Keep the original first-script Combined summary calculation unchanged.
    combined_elev=[]
    combined_azim=[]
    for item in evaluation:
        combined_elev.append(
            item["elev_mean"]
        )
        combined_azim.append(
            item["azim_mean"]
        )

    evaluation.append(
        {
        "name":"Combined",
        "elev_mean":
            float(
                np.mean(combined_elev)
            ),
        "elev_std":
            float(
                np.std(combined_elev)
            ),
        "elev_95":
            float(
                np.max(
                    [
                    x["elev_95"]
                    for x in evaluation
                    ]
                )
            ),
        "azim_mean":
            float(
                np.mean(combined_azim)
            ),
        "azim_std":
            float(
                np.std(combined_azim)
            ),
        "azim_95":
            float(
                np.max(
                    [
                    x["azim_95"]
                    for x in evaluation
                    ]
                )
            )
        }
    )

    sensitivity_data["Combined"]=(
        np.asarray(all_elev_errors,dtype=float),
        np.asarray(all_azim_errors,dtype=float)
    )

    save_alignment_evaluation(
        evaluation,
        sensitivity_data
    )
    print(
        "\nStep 6: Generate matrices"
    )
    for const_name,cfg in CONSTELLATIONS.items():
        for freq_label in cfg["freqs"]:
            for feature in FEATURES:
                print(
                    "Generating:",
                    const_name,
                    freq_label,
                    feature
                )
                matrix=generate_feature_matrix(
                    const_name,
                    freq_label,
                    feature,
                    source_list,
                    selected_mapping
                )
                save_feature_matrix(
                    const_name,
                    freq_label,
                    feature,
                    matrix,
                    offsets,
                    counter
                )
    print(
        "\nFinished."
    )

if __name__=="__main__":

    main()