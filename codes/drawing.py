import pandas as pd
import os

try:
    from IPython.display import SVG, display
    in_notebook = True
except ImportError:
    SVG = None
    display = None
    in_notebook = False

try:
    import cairosvg
    has_cairosvg = True
except ImportError:
    cairosvg = None
    has_cairosvg = False

def render_svgs(csv_path='predictions.csv'):
    """
    Reads a CSV file with columns 'id', 'description', and 'svg',
    saves each SVG to a separate file, optionally converts to PNG,
    and displays inline in a Jupyter environment.
    """
    df = pd.read_csv(csv_path)
    os.makedirs('svgs', exist_ok=True)
    if has_cairosvg:
        os.makedirs('pngs', exist_ok=True)

    for _, row in df.iterrows():
        file_id = row['id']
        svg_content = row['svg']

        # Save SVG file
        svg_file = os.path.join('svgs', f"{file_id}.svg")
        with open(svg_file, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        print(f"Saved SVG: {svg_file}")

        # Convert to PNG if cairosvg is available
        if has_cairosvg:
            png_file = os.path.join('pngs', f"{file_id}.png")
            cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), write_to=png_file)
            print(f"Converted to PNG: {png_file}")

        # Display inline if in a Jupyter notebook
        if in_notebook and SVG and display:
            display(SVG(filename=svg_file))

if __name__ == "__main__":
    render_svgs()

