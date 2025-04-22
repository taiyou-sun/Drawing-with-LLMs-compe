import argparse
import csv
import logging
import pandas as pd
import torch
import cairosvg
import re
import re2
from lxml import etree
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import kagglehub

#| export
import concurrent

# Import the SVG constraints package
svg_constraints = kagglehub.package_import('metric/svg-constraints')

# Define device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Import and download the Gemma model via kagglehub
package = kagglehub.package_import('ryanholbrook/drawing-with-llms-getting-started-with-gemma-2/versions/9')

class Model:
    def __init__(self):
        # Quantization Configuration
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        # Download model
        self.model_path = kagglehub.model_download('google/gemma-2/Transformers/gemma-2-9b-it/2')
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map={"": 0},
            quantization_config=quantization_config,
        )
        # Prompt template and defaults
        self.prompt_template = """Generate SVG code to visually represent the following text description, while respecting the given constraints.
<constraints>
* **Allowed Elements:** `svg`, `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, `defs`
* **Allowed Attributes:** `viewBox`, `width`, `height`, `fill`, `stroke`, `stroke-width`, `d`, `cx`, `cy`, `r`, `x`, `y`, `rx`, `ry`, `x1`, `y1`, `x2`, `y2`, `points`, `transform`, `opacity`
</constraints>

<example>
<description>"A red circle with a blue square inside"</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
  <circle cx="50" cy="50" r="40" fill="red"/>
  <rect x="30" y="30" width="40" height="40" fill="blue"/>
</svg>
```
</example>


Please ensure that the generated SVG code is well-formed, valid, and strictly adheres to these constraints. Always give the complete SVG code with nothing omitted. Never use an ellipsis.

<description>"{}"</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
"""
        self.default_svg = """<svg width=\"256\" height=\"256\" viewBox=\"0 0 256 256\"><circle cx=\"50\" cy=\"50\" r=\"40\" fill=\"red\" /></svg>"""
        self.constraints = svg_constraints.SVGConstraints()
        self.timeout_seconds = 90

    def predict(self, description: str, max_new_tokens=512) -> str:
        def generate_svg():
            try:
                prompt = self.prompt_template.format(description)
                inputs = self.tokenizer(text=prompt, return_tensors="pt").to(DEVICE)

                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                    )

                output_decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
                matches = re.findall(r"<svg.*?</svg>", output_decoded, re.DOTALL | re.IGNORECASE)
                svg = matches[-1] if matches else self.default_svg
                svg = self.enforce_constraints(svg)
                # Validate via cairosvg
                cairosvg.svg2png(bytestring=svg.encode('utf-8'))
                return svg
            except Exception as e:
                logging.error('SVG generation error: %s', e)
                return self.default_svg

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(generate_svg)
            try:
                return future.result(timeout=self.timeout_seconds)
            except concurrent.futures.TimeoutError:
                logging.warning("Timed out after %s seconds", self.timeout_seconds)
                return self.default_svg
            except Exception as e:
                logging.error("Unexpected error in predict: %s", e)
                return self.default_svg

    def enforce_constraints(self, svg_string: str) -> str:
        parser = etree.XMLParser(remove_blank_text=True, remove_comments=True)
        try:
            root = etree.fromstring(svg_string, parser=parser)
        except Exception:
            return self.default_svg

        # Remove disallowed elements and attributes
        to_remove = []
        for elem in root.iter():
            tag = etree.QName(elem.tag).localname
            if tag not in self.constraints.allowed_elements:
                to_remove.append(elem)
                continue
            remove_attrs = [attr for attr in elem.attrib if etree.QName(attr).localname not in self.constraints.allowed_elements[tag] and etree.QName(attr).localname not in self.constraints.allowed_elements['common']]
            for attr in remove_attrs:
                del elem.attrib[attr]
        for elem in to_remove:
            if elem.getparent() is not None:
                elem.getparent().remove(elem)

        try:
            return etree.tostring(root, encoding='unicode')
        except Exception:
            return self.default_svg


def run_predictions(input_csv: str, output_csv: str, log_file: str):
    # Configure logging
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    logging.info("Loading test data from %s", input_csv)
    # Load test data
    df = pd.read_csv(input_csv)
    model = Model()
    records = []
    for _, row in df.iterrows():
        desc = row.get('description') or row.get('prompt') or ''
        svg = model.predict(desc)
        records.append({'id': row.get('id', ''), 'description': desc, 'svg': svg})
        logging.info("Predicted SVG for id=%s", row.get('id', ''))

    # Write predictions to CSV
    logging.info("Writing predictions to %s", output_csv)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'description', 'svg'])
        writer.writeheader()
        writer.writerows(records)
    logging.info("Completed writing %d records", len(records))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run GEMMA-2 predictions on test data')
    parser.add_argument('--input', type=str, required=True, help='Path to test CSV with description column')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output CSV path')
    parser.add_argument('--log', type=str, default='predict.log', help='Log file path')
    args = parser.parse_args()
    run_predictions(args.input, args.output, args.log)
