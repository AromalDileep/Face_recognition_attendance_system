import serial
import serial.tools.list_ports
import time
import logging

def find_arduino_port():
    """
    Finds the COM port for Arduino Uno based on VID:PID (2341:0043).
    """
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "2341:0043" in port.hwid:
            return port.device
    return None

def trigger_motor(port=None, baud=9600):
    """
    Triggers the motor via serial communication.
    If port is None, it attempts to auto-detect the Arduino Uno.
    Sends "START\n" to the specified port.
    """
    if port is None:
        port = find_arduino_port()

    if port:
        try:
            print(f"Arduino Uno detected on {port}")
            logging.info(f"Arduino Uno detected on {port}")
            with serial.Serial(port, baud, timeout=1) as ser:
                time.sleep(2)            # Allow Arduino to reset
                ser.write(b"START\n")    # Send command
                ser.flush()              # Ensure transmission
                print("Motor cycle triggered")
                logging.info(f"Sent START command to {port}")
        except serial.SerialException as e:
            print(f"Error triggering motor on {port}: {e}")
            logging.error(f"Failed to trigger motor: {e}")
        except Exception as e:
            print(f"Unexpected error in trigger_motor: {e}")
            logging.error(f"Unexpected error in trigger_motor: {e}")
    else:
        print("Arduino not found.")
        logging.warning("Arduino not found.")
