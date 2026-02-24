import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import os
import time
import logging
from utils.storage import load_embeddings

PERIOD_SHEETS = {
    "Period-1": "1OA1YZiZ2FdvEkJapimsoYy8mKe-jWSMj5uidKlMKeJk",
    "Period-2": "18C1gpriQ-hIMX0yilMim-sG-qqqqO3nMdURTHS3ZMZU",
    "Period-3": "1Wmxn-OkFvs8dui-4tmNBzJzkuTkVdKNyPk50rmbkJVk",
    "Period-4": "1bBXPaLpvKopG5Ol9O3QOpx7wFC9-UQkxB2m6V1kzaqU",
    "Period-5": "1_uOgYjx4rKnFU6WWbei8881MVAC1mbJjZvqVcNLNoVE",
    "Period-6": "183RtCmveFRXZRs8z4cvaergYGxCZFAh-Fj7gP4iQiQw",
}

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
        self.client = gspread.authorize(creds)

        # Initialize with None, wait for start_session
        self.sheet = None
        self.today_str = None

    def start_session(self, period_name):
        """
        Switch to a specific worksheet based on date (e.g., 'February-First')
        period_name should match keys in PERIOD_SHEETS (e.g., 'Period-1')
        """
        sheet_id = PERIOD_SHEETS.get(period_name)
        if not sheet_id:
            logging.error(f"Invalid period name: {period_name}")
            return False

        try:
            spreadsheet = self.client.open_by_key(sheet_id)
            
            now = datetime.now()
            month_name = now.strftime("%B")  # e.g. "January"
            suffix = "First" if now.day <= 15 else "Second"
            sheet_name = f"{month_name}-{suffix}"
            
            self.sheet = spreadsheet.worksheet(sheet_name)
            self.today_str = now.strftime("%d-%m-%Y")
            logging.info(f"Switched to sheet: {sheet_name} in {period_name}")
            return True
        except Exception as e:
            logging.error(f"Error starting session: {e}")
            return False

    def can_mark(self, student_name):
        """
        Check cooldown
        """
        now = time.time()
        last = self.last_marked.get(student_name, 0)
        return (now - last) >= self.cooldown

    def mark_attendance(self, student_name):
        """
        Write attendance to Google Sheet using update_cell.
        If the student is enrolled but not in the sheet, add them at the bottom.
        """
        if self.sheet is None or not self.today_str:
            logging.warning("No session started! Cannot mark attendance.")
            return False

        if not self.can_mark(student_name):
            return False

        try:
            all_values = self.sheet.get_all_values()
            
            # Find today's column (Row 2, index 1)
            if len(all_values) < 2:
                logging.warning("Sheet does not have complete headers (Row 2 missing).")
                return False
                
            headers = all_values[1]
            if self.today_str not in headers:
                logging.warning(f"Today's date ({self.today_str}) not found in headers.")
                return False
                
            col_idx = headers.index(self.today_str) + 1  # 1-based index for gspread
            
            # Find student row (Row 3 onwards)
            student_row_idx = None
            for row_num, row_data in enumerate(all_values[2:], start=3):  # 1-based start at row 3
                if len(row_data) >= 2 and row_data[1].strip() == student_name.strip():
                    student_row_idx = row_num
                    break
                    
            if student_row_idx is None:
                # Student not in sheet. Since they were recognized, they are enrolled.
                logging.info(f"Student {student_name} not found in sheet. Adding them...")
                # We need to append a new row
                new_row_idx = len(all_values) + 1
                
                # Make a row matching the width of headers
                new_row = [""] * len(headers)
                new_row[0] = "=ROW()-2"
                new_row[1] = student_name
                new_row[col_idx - 1] = "P"  # col_idx is 1-based, list is 0-based
                
                self.sheet.append_row(
                    new_row, 
                    table_range=f"A{new_row_idx}",
                    value_input_option="USER_ENTERED"
                )
                student_row_idx = new_row_idx
            else:
                self.sheet.update_cell(student_row_idx, col_idx, "P")
            
            self.last_marked[student_name] = time.time()
            logging.info(f"Attendance marked for {student_name}")
            return True
        except Exception as e:
            logging.error(f"Error marking attendance: {e}")
            return False

    def mark_absent_after_session(self):
        """
        Mark students absent (AB) if enrolled but unmarked,
        or Not Enrolled (NA) if in sheet but not enrolled.
        If a student is enrolled but not in the sheet, they are added and marked AB.
        """
        if self.sheet is None or not self.today_str:
            logging.warning("No session started! Cannot process absences.")
            return False
            
        try:
            all_values = self.sheet.get_all_values()
            
            if len(all_values) < 2:
                return False
                
            headers = all_values[1]
            if self.today_str not in headers:
                logging.warning(f"Today's date ({self.today_str}) not found in headers to mark absences.")
                return False
                
            col_idx = headers.index(self.today_str) + 1
            col_idx_0_based = col_idx - 1 
            
            enrolled_students = list(load_embeddings().keys())
            
            # Map students in sheet
            sheet_students = {}  # {student_name: row_num}
            sl_nos = []
            
            for row_num, row_data in enumerate(all_values[2:], start=3):
                if len(row_data) >= 1 and str(row_data[0]).isdigit():
                    sl_nos.append(int(row_data[0]))
                    
                if len(row_data) >= 2:
                    student_name = row_data[1].strip()
                    if student_name:
                        sheet_students[student_name] = (row_num, row_data)
            
            next_sl_no = max(sl_nos) + 1 if sl_nos else 1
            new_row_idx = len(all_values) + 1
            
            # First, check for enrolled students NOT in the sheet
            for student_name in enrolled_students:
                if student_name not in sheet_students:
                    # Add them to the sheet and mark AB
                    logging.info(f"Adding absent enrolled student {student_name} to sheet.")
                    new_row = [""] * len(headers)
                    new_row[0] = "=ROW()-2"
                    new_row[1] = student_name
                    new_row[col_idx_0_based] = "AB"
                    
                    self.sheet.append_row(
                        new_row, 
                        table_range=f"A{new_row_idx}",
                        value_input_option="USER_ENTERED"
                    )
                    
                    # Update local trackers so we don't process them again
                    sheet_students[student_name] = (new_row_idx, new_row)
                    new_row_idx += 1
            
            # Now, process everyone in the sheet
            for student_name, (row_num, row_data) in sheet_students.items():
                cell_value = ""
                if len(row_data) > col_idx_0_based:
                    cell_value = row_data[col_idx_0_based].strip()
                    
                if cell_value in ("P", "AB", "NA"):
                    continue
                    
                if not cell_value:
                    if student_name in enrolled_students:
                        self.sheet.update_cell(row_num, col_idx, "AB")
                        
            logging.info("Absent marking complete.")
            return True
        except Exception as e:
            logging.error(f"Error marking absences: {e}")
            return False
