import serial
import time
import logging

def trigger_motor(port="COM3", baud=9600):
    """
    Triggers the motor via serial communication.
    Sends "START\n" to the specified port.
    """
    try:
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
