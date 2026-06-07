"""
This script converts raw u-blox UBX JSON messages into hourly observation, satelliteInformation, and pvtSolution JSON files.
"""

import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent

RAW_ROOT = PROJECT_ROOT / "RawData"
SAVE_ROOT = PROJECT_ROOT / "Processed_Data"

DAYS = ["20260520"]
HOURS = range(13, 24)

RUN_RAWX = True
RUN_SAT = True
RUN_PVT = True


CONSTELLATIONS = [
    {"name": "G", "gnss_id": 0, "num_sats": 32, "sig_id_2": 3},
    {"name": "E", "gnss_id": 2, "num_sats": 36, "sig_id_2": 6},
    {"name": "B", "gnss_id": 3, "num_sats": 63, "sig_id_2": 2},
    {"name": "Q", "gnss_id": 5, "num_sats": 10, "sig_id_2": 5},
    {"name": "R", "gnss_id": 6, "num_sats": 33, "sig_id_2": 2},
]

RAWX_FIELDS = [
    ("prMes", "prMes"),
    ("doMes", "doMes"),
    ("cpMes", "cpMes"),
    ("cno", "cn0"),
    ("prStd", "prStd"),
    ("cpStd", "cpStd"),
    ("doStd", "doStd"),
]

SAT_FIELDS = [
    ("svId", "svId", 0.0),
    ("svUsed", "svUsed", 0.11),
    ("cno", "cno", 0.0),
    ("elev", "elev", 0.0),
    ("azim", "azim", 0.0),
    ("prRes", "prRes", 0.0),
    ("qualityInd", "qualityInd", 0.11),
    ("health", "health", 0.11),
]

PVT_FIELDS = [
    "numSV", "nano", "lon", "lat", "height", "velN", "velE", "velD",
    "hMSL", "hAcc", "vAcc", "sAcc", "gSpeed", "headMot", "headAcc",
]

POSECEF_FIELDS = ["ecefX", "ecefY", "ecefZ"]
CLOCK_FIELDS = ["clkB", "clkD", "tAcc", "fAcc"]
DOP_FIELDS = ["gDOP", "pDOP", "tDOP", "vDOP", "hDOP", "nDOP", "eDOP"]


def key_suffix(index: int) -> str:
    return f"0{index}" if index < 10 else str(index)


def to_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_hour_dir(day: str, hour: int) -> Path:
    day_dir = RAW_ROOT / str(day)
    hour_dir_1 = day_dir / str(hour)
    hour_dir_2 = day_dir / f"{int(hour):02d}"

    if hour_dir_1.exists():
        return hour_dir_1

    if hour_dir_2.exists():
        return hour_dir_2

    return hour_dir_1


def get_message_dir(day: str, hour: int, message_name: str) -> Path:
    return get_hour_dir(day, hour) / message_name


def list_json_files(folder: Path):
    if not folder.is_dir():
        print(f"Skip missing folder: {folder}")
        return []

    return sorted(folder.glob("*.json"))


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def get_sat_index(content: dict, suffix: str, num_sats: int):
    sv_id = to_int(content.get(f"svId_{suffix}"))

    if sv_id is None:
        return None

    index = sv_id - 1

    if 0 <= index < num_sats:
        return index

    return None


def extract_rawx_by_constellation(content: dict, cfg: dict):
    num_sats = cfg["num_sats"]
    result = {"VS": np.zeros((1, num_sats))}

    for _, out_name in RAWX_FIELDS:
        result[f"{out_name}_1"] = np.zeros((1, num_sats))
        result[f"{out_name}_2"] = np.zeros((1, num_sats))

    num_meas = to_int(content.get("numMeas"), 0)

    for index in range(1, num_meas + 1):
        suffix = key_suffix(index)

        if to_int(content.get(f"gnssId_{suffix}")) != cfg["gnss_id"]:
            continue

        sat_index = get_sat_index(content, suffix, num_sats)

        if sat_index is None:
            continue

        result["VS"][0, sat_index] = content.get(f"svId_{suffix}", 0)

        sig_id = to_int(content.get(f"sigId_{suffix}"))

        if sig_id == 0:
            band = 1
        elif sig_id == cfg["sig_id_2"]:
            band = 2
        else:
            continue

        for src_name, out_name in RAWX_FIELDS:
            result[f"{out_name}_{band}"][0, sat_index] = content.get(
                f"{src_name}_{suffix}", 0
            )

    return {key: value.tolist() for key, value in result.items()}


def init_rawx_data():
    data = {"recordTime": []}

    for cfg in CONSTELLATIONS:
        data[f"VS{cfg['name']}"] = []

    for cfg in CONSTELLATIONS:
        name = cfg["name"]

        for _, out_name in RAWX_FIELDS:
            data[f"{out_name}_{name}1"] = []
            data[f"{out_name}_{name}2"] = []

    return data


def process_rawx(days, hours):
    for day in days:
        for hour in hours:
            folder = get_message_dir(day, hour, "RXM-RAWX")
            files = list_json_files(folder)

            if not files:
                continue

            data = init_rawx_data()

            for file_path in files:
                print(f"Reading: {file_path}")
                content = load_json(file_path)

                data["recordTime"].append(content.get("start_time"))

                for cfg in CONSTELLATIONS:
                    name = cfg["name"]
                    extracted = extract_rawx_by_constellation(content, cfg)

                    data[f"VS{name}"].extend(extracted["VS"])

                    for _, out_name in RAWX_FIELDS:
                        data[f"{out_name}_{name}1"].extend(extracted[f"{out_name}_1"])
                        data[f"{out_name}_{name}2"].extend(extracted[f"{out_name}_2"])

            out_path = SAVE_ROOT / str(day) / f"observation{hour}.json"
            save_json(out_path, data)
            print(f"Saved: {out_path}")


def extract_sat_by_constellation(content: dict, cfg: dict):
    num_sats = cfg["num_sats"]
    result = {}

    for _, out_name, default_value in SAT_FIELDS:
        result[out_name] = np.full((1, num_sats), default_value)

    num_svs = to_int(content.get("numSvs"), 0)

    for index in range(1, num_svs + 1):
        suffix = key_suffix(index)

        if to_int(content.get(f"gnssId_{suffix}")) != cfg["gnss_id"]:
            continue

        sat_index = get_sat_index(content, suffix, num_sats)

        if sat_index is None:
            continue

        for src_name, out_name, _ in SAT_FIELDS:
            result[out_name][0, sat_index] = content.get(f"{src_name}_{suffix}", 0)

    return {key: value.tolist() for key, value in result.items()}


def init_sat_data():
    data = {"recordTime": [], "numSvs": []}

    for cfg in CONSTELLATIONS:
        name = cfg["name"]

        for _, out_name, _ in SAT_FIELDS:
            data[f"{out_name}_{name}"] = []

    return data


def process_satellite_information(days, hours):
    for day in days:
        for hour in hours:
            folder = get_message_dir(day, hour, "NAV-SAT")
            files = list_json_files(folder)

            if not files:
                continue

            data = init_sat_data()

            for file_path in files:
                print(f"Reading: {file_path}")
                content = load_json(file_path)

                data["recordTime"].append(content.get("start_time"))
                data["numSvs"].append(content.get("numSvs"))

                for cfg in CONSTELLATIONS:
                    name = cfg["name"]
                    extracted = extract_sat_by_constellation(content, cfg)

                    for _, out_name, _ in SAT_FIELDS:
                        data[f"{out_name}_{name}"].extend(extracted[out_name])

            out_path = SAVE_ROOT / str(day) / f"satelliteInformation{hour}.json"
            save_json(out_path, data)
            print(f"Saved: {out_path}")


def init_pvt_data():
    fields = ["recordTime"] + PVT_FIELDS + POSECEF_FIELDS + CLOCK_FIELDS + DOP_FIELDS
    return {field: [] for field in fields}


def append_fields_from_folder(
    data: dict,
    day: str,
    hour: int,
    message_name: str,
    fields: list,
    append_record_time: bool = False,
):
    folder = get_message_dir(day, hour, message_name)
    files = list_json_files(folder)

    for file_path in files:
        print(f"Reading: {file_path}")
        content = load_json(file_path)

        if append_record_time:
            data["recordTime"].append(content.get("start_time"))

        for field in fields:
            data[field].append(content.get(field))


def process_pvt_solution(days, hours):
    for day in days:
        for hour in hours:
            data = init_pvt_data()

            append_fields_from_folder(
                data=data,
                day=day,
                hour=hour,
                message_name="NAV-PVT",
                fields=PVT_FIELDS,
                append_record_time=True,
            )

            append_fields_from_folder(
                data=data,
                day=day,
                hour=hour,
                message_name="NAV-POSECEF",
                fields=POSECEF_FIELDS,
            )

            append_fields_from_folder(
                data=data,
                day=day,
                hour=hour,
                message_name="NAV-CLOCK",
                fields=CLOCK_FIELDS,
            )

            append_fields_from_folder(
                data=data,
                day=day,
                hour=hour,
                message_name="NAV-DOP",
                fields=DOP_FIELDS,
            )

            if not any(data.values()):
                print(f"Skip empty data: day={day}, hour={hour}")
                continue

            out_path = SAVE_ROOT / str(day) / f"pvtSolution{hour}.json"
            save_json(out_path, data)
            print(f"Saved: {out_path}")


def main():
    if RUN_RAWX:
        process_rawx(DAYS, HOURS)

    if RUN_SAT:
        process_satellite_information(DAYS, HOURS)

    if RUN_PVT:
        process_pvt_solution(DAYS, HOURS)


if __name__ == "__main__":
    main()