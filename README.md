# Face Recognition Attendance System

A Windows-based application for automated attendance tracking using facial recognition. The system integrates with Google Sheets for logging attendance and communicates with an Arduino microcontroller to signal the start and stop of attendance sessions.

## Project Structure Setup

To run this project, you need to set up a few essential files in their respective directories:

### 1. Credentials

You will need a Google Service Account to interact with Google Sheets.

1. Generate your service account key from the Google Cloud Console.
2. Download the JSON file.
3. Rename it to `service_account.json` and place it in the `credentials/` folder.

### 2. Face Data

The system requires an `embeddings.pkl` file which contains the encoded face data for the registered users.

- Place your `embeddings.pkl` file inside the `data/` folder.

## Hardware Integration (Arduino)

The application communicates via serial port with an Arduino (specifically the **Arduino Uno r3** model).

- When an attendance session is started in the app, it sends a `start` signal to the Arduino.
- When the session is stopped, it sends a `stop` signal to the Arduino.

## Usage

1. Ensure your Arduino Uno r3 is connected.
2. Make sure your `credentials/service_account.json` and `data/embeddings.pkl` files are in place.
3. Run the application on your Windows machine.
4. Use the interface to start the attendance session (which signals the Arduino and begins facial recognition).
5. Stop the session when finished.
