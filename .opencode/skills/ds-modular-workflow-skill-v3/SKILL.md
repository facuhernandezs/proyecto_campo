---
name: ds-modular-workflow-skill-v3
description: performing any Data Science project stage (EDA, Preprocessing, Feature Engineering, Modeling, Evaluation) to enforce a clean, decoupled, and modular code architecture.
---

## 1. Architectural Rules & Separation of Concerns

- **NO "Spaghetti Code" in Notebooks**: All heavy processing logic, statistical computations, data transformations, feature engineering, and model training routines MUST reside in reusable Python modules (`.py` files) inside the source directory (e.g., `src/` or the project's designated modular code folder).
- **Lean & Narrative Notebooks**: Jupyter Notebooks (`.ipynb` files in `notebooks/` or stage-specific folders) are reserved exclusively for narrative flow, sequential execution of imported `.py` functions, visual plots, and executive summaries.
- **Flexible & Project-Driven Structure**: The exact file names and directory layout can adapt based on project requirements or the Specification Document (SDD). The core rule is strictly separating executable logic (`.py`) from presentation/experimentation (`.ipynb`).

---

## 2. Step-by-Step Execution Workflow

### Step 1: Define or Update the Python Module (`.py`)
Before creating or modifying a notebook for any project stage, write the corresponding functions in the appropriate `.py` module inside `src/`. Include type hints and clean docstrings.

**Example in `src/preprocessing.py` (or stage-relevant module):**
```python
import pandas as pd
import numpy as np

def clean_and_impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans column headers, handles data types, and imputes missing values 
    according to project specifications.
    """
    df_clean = df.copy()
    df_clean.columns = [c.strip().lower().replace(" ", "_") for c in df_clean.columns]
    
    # Numeric median imputation
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
    return df_clean
```

### Step 2: Invoke Functions from the Jupyter Notebook (`.ipynb`)
In the corresponding notebook for the current stage, configure the project root path and import the modular functions. Avoid inline function definitions inside notebook cells.

**Notebook Cell Layout:**

*Cell 1 (Path Setup & Modular Imports):*
```python
import sys
from pathlib import Path

# Add project root to sys.path to enable imports from src/
ROOT_DIR = Path("..").resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Import modular functions from src/
from src.preprocessing import clean_and_impute_data
```

*Cell 2 (Execution of Modular Pipeline):*
```python
df_raw = pd.read_csv(ROOT_DIR / "data" / "dataset.csv")
df_clean = clean_and_impute_data(df_raw)
df_clean.head()
```

*Cell 3 (RDD Control Metrics / Output Verification):*
```python
print(f"Processed Rows: {len(df_clean)}")
print(f"Remaining Nulls: {df_clean.isnull().sum().sum()}")
```

---

## 3. Output Checklist
- [ ] Executable logic, transformations, and algorithms reside in `.py` modules inside `src/`.
- [ ] Notebooks (`.ipynb`) remain lean with no long inline function definitions.
- [ ] Functions are correctly imported from `src/` into the notebook environment.
- [ ] Execution is verified via concise metrics/outputs (RDD receipt).
