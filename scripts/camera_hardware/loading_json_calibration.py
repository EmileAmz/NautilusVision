#!/usr/bin/env python3

from pathlib import Path
import depthai as dai


CALIBRATION_FILE = "C:/Users/Xavier Lefebvre/Documents/GitHub/NautilusVision/scripts/Annotate_And_Save/calibration.json"

# Backup automatique
BACKUP_FILE = str((Path(__file__).parent / "json_config" / "depthai_calib_backup.json").resolve())

def find_oakd_device():
    device_infos = dai.Device.getAllAvailableDevices()

    print(f"Found {len(device_infos)} device(s).")

    for device_info in device_infos:
        print(f"\nChecking device: {device_info}")

        try:
            with dai.Device(device_info) as device:
                cameras = device.getConnectedCameras()
                print(f"Connected cameras: {cameras}")

                if (
                    dai.CameraBoardSocket.CAM_B in cameras
                    and dai.CameraBoardSocket.CAM_C in cameras
                ):
                    print("OAK-D found.")
                    return device_info

        except Exception as e:
            print(f"Could not check device: {e}")

    return None


try:
    device = dai.Device(find_oakd_device())

    print("\n====================================")
    print("Connected Device")
    print("====================================")

    current_calib = device.readCalibration()
    current_calib.eepromToJsonFile(BACKUP_FILE)

    print("Current calibration backed up to:")
    print(BACKUP_FILE)
    print()

    answer = input(
        f"Flash calibration file:\n{CALIBRATION_FILE}\n\n"
        "Continue? (y/n): "
    ).strip().lower()

    if answer != "y":
        print("Operation cancelled.")
        exit(0)

    calib_data = dai.CalibrationHandler(CALIBRATION_FILE)
    print("\nFlashing calibration...")
    device.flashCalibration(calib_data)

    print("\nSuccessfully flashed calibration!")

except Exception as e:
    print("\nFailed flashing calibration:")
    print(e)