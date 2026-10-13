
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class ScribeDataset(Dataset):
    def __init__(self, data_dir, tokenizer, data_source ='ouh'):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.data = pd.read_csv(data_dir)
        if data_source == "MIMIC":
            self.data.rename(columns={"findings": "Findings", "impression":"Conclusion","subject_id":"patient_id", "reference_report":"redacted_report"}, inplace=True)
        elif data_source == "ct-rate":
            self.data.rename(columns={"Findings_EN": "Findings", "Impressions_EN": "Conclusion"}, inplace=True)
            if "patient_id" not in self.data.columns:
                for candidate in ("VolumeName", "PatientID", "subject_id"):
                    if candidate in self.data.columns:
                        self.data.rename(columns={candidate: "patient_id"}, inplace=True)
                        break
        print(len(self.data))
        self.data = self.data.dropna(subset=['Findings', 'Conclusion', 'patient_id']).reset_index()
        print(len(self.data))
        if "redacted_report" not in self.data.columns:
            self.data["redacted_report"] = self.data["Findings"].fillna("") + " " + self.data["Conclusion"].fillna("")
        #self.data = self.data[self.data['Findings'].notna() & self.data['Conclusion'].notna() & self.data['patient_id'].notna()]
        self.reports = self.data['redacted_report']
        self.findings = self.data['Findings']
        self.conclusion = self.data['Conclusion']
        self.ids = self.data['patient_id']
        
        return

    def __len__(self):
        return len(self.ids)
    
    def no_of_reports_overx_tokens(self, no_tokens):
        count = 0
        for report in self.reports:
            if len(self.tokenizer.encode(report)) > no_tokens:
                count += 1
        
        return count

    def __getitem__(self, idx):
        #input_ids = self.tokenizer.encode(self.reports[idx])
        #make an attention mask
        #attention_mask = [1]*len(input_ids)
        
        #add a BOS to the beginning and end of each report
        #report = "[BOS]" + self.reports[idx] + "[BOS]"
        findings = "[BOS]" + "findings: " + self.findings[idx] + " conclusion: " #removed \n from after findings and before conclusions before using wp
        conclusion = self.conclusion[idx] + "[BOS]"
        report = findings + conclusion

        sample = {
            #'input_ids': input_ids,
            #'attention_mask': attention_mask,
            'ids': self.ids[idx],
            'reference_report': report, #self.reports[idx]
            'findings': findings,
            'conclusion': conclusion
        }

        #print("sample")
        #print(sample)
        return sample