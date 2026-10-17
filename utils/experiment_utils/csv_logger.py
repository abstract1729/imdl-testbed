from datetime import datetime

import pandas as pd


class CSVLogger:
    """
    Utility class for logging experiment outputs.

    One instance corresponds to one experiment run.
    """

    def __init__(self, output_csv):
        self.output_csv = output_csv
        self.run_timestamp = datetime.now().strftime( "%Y-%m-%d %H:%M:%S")
        self.results = []

    # ------------------------------------------------------------

    def log_success(self,image_id,file_name,label,image_path,image_height,
                    image_width,score,inference_time_ms):

        self.results.append(
            {
                "run_timestamp": self.run_timestamp,
                "image_id": image_id,
                "file_name": file_name,
                "label": label,
                "image_path": image_path,
                "image_height": image_height,
                "image_width": image_width,
                "score": score,
                "inference_time_ms": inference_time_ms,
                "inference_status": "SUCCESS",
                "error_type": "",
                "error_message": "",
            }
        )

    # ------------------------------------------------------------

    def log_failure(self,image_id,file_name,label,image_path,image_height=None,
                    image_width=None,error_type="",error_message=""):

        self.results.append(
            {
                "run_timestamp": self.run_timestamp,
                "image_id": image_id,
                "file_name": file_name,
                "label": label,
                "image_path": image_path,
                "image_height": image_height,
                "image_width": image_width,
                "score": None,
                "inference_time_ms": None,
                "inference_status": "FAILED",
                "error_type": error_type,
                "error_message": str(error_message),
            }
        )

    # ------------------------------------------------------------

    def save(self):
        df = pd.DataFrame(
            self.results,
            columns=["run_timestamp","image_id","file_name","label","image_path","image_height",
                     "image_width","score","inference_time_ms","inference_status","error_type","error_message",])

        df.to_csv(self.output_csv,index=False)
        return self.output_csv