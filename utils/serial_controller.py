import serial
import serial.tools.list_ports
import time
import logging

def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "Arduino" in port.description or "USB-SERIAL" in port.description:
            return port.device
    return None

def send_start_signal():
    port = find_arduino_port()
    if port is None:
        print("Arduino not found.")
        logging.warning("Arduino not found.")
    else:
        print(f"Arduino found on {port}")
        logging.info(f"Arduino found on {port}")
        try:
            with serial.Serial(port, 9600, timeout=1) as ser:
                time.sleep(2)  # Allow reset
                ser.write(b"START\n")
                ser.flush()
                print("Attendance started command sent.")
                logging.info("Sent START command")
        except Exception as e:
            print(f"Error sending START: {e}")
            logging.error(f"Error sending START: {e}")

def send_stop_signal():
    port = find_arduino_port()
    if port is None:
        print("Arduino not found.")
        logging.warning("Arduino not found.")
    else:
        print(f"Arduino found on {port}")
        logging.info(f"Arduino found on {port}")
        try:
            with serial.Serial(port, 9600, timeout=1) as ser:
                time.sleep(2)  # Allow reset
                ser.write(b"STOP\n")
                ser.flush()
                print("Attendance stopped command sent.")
                logging.info("Sent STOP command")
        except Exception as e:
            print(f"Error sending STOP: {e}")
            logging.error(f"Error sending STOP: {e}")
