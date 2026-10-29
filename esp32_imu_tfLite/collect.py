# IMU 데이터 수집
################################
# 실행하면 data/train 폴더가 생성되고 ESP32로 데이터 수집 가능
# ESP32 코드: "D:\Documents\Arduino\esp_CNN_MPU6050"
################################

import argparse
from pathlib import Path
from uuid import uuid4

import numpy as np

from sensor_io import SensorClient, CLASS_NAMES, countdown


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ip", required=True)
    parser.add_argument("--label", choices=CLASS_NAMES, required=True)
    parser.add_argument(
        "--split", choices=["train", "val", "test"], required=True
    )
    parser.add_argument("--session", required=True)
    parser.add_argument("--count", type=int, default=20)

    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be positive.")

    # 세션 이름에 경로 문자가 들어가지 않도록 제한
    if not all(c.isalnum() or c in "-_" for c in args.session):
        parser.error("Use letters, numbers, '-' or '_' for session.")

    out_dir = (
        Path("data")
        / args.split
        / args.label
        / args.session
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    client = SensorClient(args.ip)

    try:
        saved = 0

        while saved < args.count:
            print(
                f"\nLabel={args.label}  "
                f"Split={args.split}  "
                f"Saved={saved}/{args.count}"
            )

            command = input("Enter: record / q: quit > ").strip().lower()

            if command == "q":
                break

            countdown()

            try:
                raw = client.record()
            except ValueError as error:
                print(f"Discarded: {error}")
                continue

            print("STOP.")

            answer = input("Save? Enter=yes / n=no > ").strip().lower()
            if answer == "n":
                print("Discarded.")
                continue

            filename = out_dir / f"{uuid4().hex}.csv"

            np.savetxt(
                filename,
                raw,
                delimiter=",",
                header="t_us,ax,ay,az,gx,gy,gz",
                comments="",
                fmt=["%.0f"] + ["%.6f"] * 6,
            )

            saved += 1
            print(f"Saved: {filename}")

    finally:
        client.close()


if __name__ == "__main__":
    main()