import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import os
import time


class AttendanceManager:
    def __init__(self, cooldown_seconds=60):
        """
        cooldown_seconds: prevent duplicate attendance within this time
        """
        self.cooldown = cooldown_seconds
        self.last_marked = {}  # {student_id: timestamp}

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        creds_path = os.path.join(base_dir, "credentials", "service_account.json")

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]

        creds = Credentials.from_service_account_file(
            creds_path, scopes=scopes
        )
        client = gspread.authorize(creds)

        self.spreadsheet = client.open("Face Recognition Attendance")
        # Initialize with None, wait for start_session
        self.sheet = None
        try:
             self.sheet = self.spreadsheet.worksheet("Sheet1")
        except:
             pass # Sheet1 might not exist, will be set later via start_session

    def start_session(self, sheet_name):
        """
        Switch to a specific worksheet (e.g., 'Period-1')
        """
        try:
            self.sheet = self.spreadsheet.worksheet(sheet_name)
            print(f"✅ Switched to sheet: {sheet_name}")
            return True
        except gspread.WorksheetNotFound:
            print(f"❌ Worksheet '{sheet_name}' not found!")
            return False

    def can_mark(self, student_id):
        """
        Check cooldown
        """
        now = time.time()
        last = self.last_marked.get(student_id, 0)

        return (now - last) >= self.cooldown

    def mark_attendance(self, student_id):
        """
        Write attendance to Google Sheet
        """
        if self.sheet is None:
            print("⚠️ No sheet selected! Cannot mark attendance.")
            return False

        if not self.can_mark(student_id):
            return False

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        self.sheet.append_row(
            [student_id, date_str, time_str, "Present"]
        )

        self.last_marked[student_id] = time.time()
        print(f"✅ Attendance marked for {student_id}")

        return True
