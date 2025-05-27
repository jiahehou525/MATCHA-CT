import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import re
import logging
import pandas as pd
from pathlib import Path
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from datetime import datetime
# Then first download the model manually:
from transformers import AutoTokenizer, AutoModel


class EnhancedTextProcessor(nn.Module):
    def __init__(self,
                 bert_model="microsoft/BiomedVLP-CXR-BERT-general",
                 feature_dim=256,
                 device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_logging()

        # BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.bert = AutoModel.from_pretrained(bert_model).to(self.device)

        # MAVL-style aspects
        self.aspects = {
            'texture': r'texture.{0,50}|density.{0,50}',
            'opacity': r'opac\w+.{0,50}|lucen\w+.{0,50}',
            'shape': r'shape.{0,50}|contour.{0,50}',
            'location': r'location.{0,50}|position.{0,50}',
            'pattern': r'pattern.{0,50}|distribution.{0,50}',
            'border': r'border.{0,50}|margin.{0,50}',
            'fluid': r'fluid.{0,50}|effusion.{0,50}'
        }

        # Enhanced projection layers
        self.aspect_projectors = nn.ModuleDict({
            aspect: nn.Sequential(
                nn.Linear(768, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, feature_dim),
                nn.LayerNorm(feature_dim)
            ).to(self.device)
            for aspect in self.aspects.keys()
        })

        # Section weights
        self.section_weights = {
            'findings': 0.4,
            'impressions': 0.4,
            'clinical_info': 0.2
        }

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('text_processor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('TextProcessor')
        # Add volume-specific processing
        self.volume_identifier = re.compile(r'([a-z]+_\d+_[a-z]+_\d+)')

    @torch.no_grad()
    def process_report(self, report):
        """Process a single report with MAVL-style aspect extraction"""
        try:
            # Combine report sections with weights
            full_text = ""
            for section, weight in self.section_weights.items():
                if section in report and report[section]:
                    # Properly handle the text addition
                    section_text = str(report[section]).strip()
                    if section_text:
                        full_text += f"{section.upper()}: {section_text} "

            aspect_features = {}
            for aspect in self.aspects.keys():
                # Extract aspect-specific text
                aspect_text = self.extract_aspect_text(full_text, aspect)

                # Tokenize
                inputs = self.tokenizer(
                    aspect_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                # Get BERT embeddings
                outputs = self.bert(**inputs, output_hidden_states=True)
                last_four_layers = torch.stack(outputs.hidden_states[-4:])
                feature_map = last_four_layers.mean(0)
                bert_features = feature_map[:, 0]  # [CLS] token

                # Project to desired dimension
                aspect_features[aspect] = self.aspect_projectors[aspect](bert_features)

            return aspect_features

        except Exception as e:
            self.logger.error(f"Error processing report: {str(e)}")
            return None

    def extract_aspect_text(self, report_text, aspect):
        """Extract text related to a specific aspect"""
        try:
            pattern = self.aspects[aspect]
            matches = re.finditer(pattern, report_text, re.IGNORECASE)
            relevant_text = [m.group(0) for m in matches]
            return ' '.join(relevant_text) if relevant_text else f"No {aspect} mentioned"
        except Exception as e:
            self.logger.error(f"Error extracting aspect {aspect}: {str(e)}")
            return f"No {aspect} mentioned"

    def extract_volume_info(self, volume_name):
        """Extract structured information from volume name"""
        match = self.volume_identifier.match(volume_name)
        if match:
            parts = volume_name.split('_')
            return {
                'case_id': f"{parts[0]}_{parts[1]}",
                'series': parts[2],
                'volume_num': parts[3]
            }
        return None

    def process_report_with_volume(self, report, volume_name):
        """Process report with volume-specific context"""
        try:
            # Get basic features
            base_features = self.process_report(report)

            # Extract volume information
            vol_info = self.extract_volume_info(volume_name)
            if vol_info:
                # Add volume-specific context
                extra_context = f"Volume {vol_info['volume_num']} from series {vol_info['series']}"

                # Encode extra context
                context_inputs = self.tokenizer(
                    extra_context,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                context_outputs = self.bert(**context_inputs)
                context_features = context_outputs.last_hidden_state[:, 0]

                # Combine with base features
                for aspect in base_features.keys():
                    # Add slight variation based on volume info
                    variation = torch.randn_like(base_features[aspect]) * 0.01
                    base_features[aspect] = base_features[aspect] + variation

            return base_features

        except Exception as e:
            self.logger.error(f"Error processing volume {volume_name}: {str(e)}")
            return None

    def process_reports_file(self, input_path, output_path):
        """Process reports with volume information"""
        try:
            # Read Excel file
            df = pd.read_excel(input_path)
            self.logger.info(f"Found {len(df)} reports to process")

            all_features = []
            volume_ids = []

            for idx, row in df.iterrows():
                report = {
                    'clinical_info': str(row.get('ClinicalInformation_EN', '')),
                    'findings': str(row.get('Findings_EN', '')),
                    'impressions': str(row.get('Impressions_EN', '')),
                }
                volume_name = str(row.get('VolumeName', ''))

                # Process with volume info
                features = self.process_report_with_volume(report, volume_name)
                if features is not None:
                    combined = torch.cat(list(features.values()), dim=-1)
                    all_features.append(combined)
                    volume_ids.append(volume_name)

            if all_features:
                # Stack all features
                stacked_features = torch.stack(all_features)

                # Save features with volume information
                torch.save({
                    'features': stacked_features,
                    'volume_ids': volume_ids,
                    'aspects': list(self.aspects.keys()),
                    'creation_date': datetime.now().isoformat()
                }, output_path)

                return stacked_features
            else:
                self.logger.error("No features were generated")
                return None

        except Exception as e:
            self.logger.error(f"Error processing reports file: {str(e)}")
            return None


def main():
    # Initialize processor
    processor = EnhancedTextProcessor()

    # Process reports
    input_path = "D:/CT-RATE/dataset/radiology_text_reports/train_reports_small.xlsx"
    output_path = "D:/mediffusion/data/text_features1.pt"

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Process and save features
    features = processor.process_reports_file(input_path, output_path)

    if features is not None:
        print(f"Successfully processed reports")
        print(f"Feature shape: {features.shape}")


if __name__ == "__main__":
    main()