import torch

class CustomCollator:
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.max_length = args.max_length

    def __call__(self, batch: list[dict[str]]):
        """
        batch is a list of dicts where each dict corresponds to a single image and has the keys:
          - image
          - input_ids
          - attention_mask
          - reference_report
        """
        # discard samples from batch where __getitem__ from custom_dataset failed (i.e. returned None)
        batch = list(filter(lambda x: x is not None, batch))  # filter out samples that are None
        reference_reports = []
        ids = []
        findings = []
        conclusions = []


        for i, sample_dict in enumerate(batch):
            reference_reports.append(sample_dict.pop("reference_report"))
            ids.append(sample_dict.pop("ids"))
            findings.append(sample_dict.pop("findings"))
            conclusions.append(sample_dict.pop("conclusion"))
            
        encoded_inputs = self.tokenizer(reference_reports, padding="longest", return_tensors="pt", truncation =True, return_attention_mask=True, max_length=self.max_length)
        encoded_findings = self.tokenizer(findings, padding="longest", return_tensors="pt", truncation =True, return_attention_mask=True, max_length=self.max_length)
        
        batch = {
            "input_ids": encoded_inputs["input_ids"],
            "attention_mask": encoded_inputs["attention_mask"]
        }
        batch['findings_ids'] = encoded_findings["input_ids"]
        batch['findings_msk'] = encoded_findings["attention_mask"]
        batch["conclusion"] = conclusions
        batch["reference_report"] = reference_reports
        batch["pat_ids"] = ids
        batch["findings"] = findings

        #other way
        """ reference_reports = batch["reference_report"]
        findings = batch["findings"]
        encoded_inputs = self.tokenizer(reference_reports, padding="longest", return_tensors="pt", truncation =True, return_attention_mask=True, max_length=self.max_length)
        encoded_findings = self.tokenizer(findings, padding="longest", return_tensors="pt", truncation =True, return_attention_mask=True, max_length=self.max_length)
        
        new_batch ={
            "input_ids": encoded_inputs["input_ids"],
            "attention_mask": encoded_inputs["attention_mask"]
        }
        new_batch["findings"] = encoded_findings["input_ids"]
        new_batch["conclusion"] = batch["conclusion"]
        new_batch["reference_report"] = reference_reports
        new_batch["pat_ids"] = batch["pat_ids"] """

        return batch


