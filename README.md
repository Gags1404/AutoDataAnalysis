# Data Analysis Autolysis

A comprehensive data analysis tool with LLM integration for automated insights generation.

## Features

- Automated data loading and preprocessing
- Basic statistical analysis
- Advanced machine learning insights
- Visualization generation
- Natural language reporting via DeepSeek AI

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/data-analysis-autolysis.git
cd data-analysis-autolysis
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/autolysis.py data/your_dataset.csv
```

Set your DeepSeek API token as environment variable:
```bash
export AIPROXY_TOKEN="your_api_token_here"
```

## Requirements

- Python 3.8+
- See requirements.txt for dependencies

## Project Structure

```
src/
    autolysis.py - Main analysis script
data/ - Example datasets (ignored by git)
docs/ - Generated analysis reports
```

## License

MIT
