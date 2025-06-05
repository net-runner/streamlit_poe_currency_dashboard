No problem, I'll adjust the README to reflect that project structure and simplify the setup!

---

# Path of Exile I currency dashboard

Streamlit dashboard with currency data from PoE challenge leagues (Ultimatum-Necropolis).

---

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have the following installed on your system:

* **Python 3.7 or higher:** You can download it from [python.org](https://www.python.org/downloads/).
* **pip (Python package installer):** Usually comes pre-installed with Python.

---

### Installation

1.  **Clone the repository (if applicable):**
    If your Streamlit project is in a Git repository, clone it to your local machine:
    ```bash
    git clone <repository_url>
    cd <your_project_directory>
    ```
    If you just have the files, navigate to your project directory.

2.  **Install dependencies:**
    Navigate to your project directory and install the required Python packages from the **`requirements.txt`** file:
    ```bash
    pip install -r requirements.txt
    ```

---

### Running the Project

Once the prerequisites are met and dependencies are installed, you can run your Streamlit application.

1.  **Navigate to the project directory:**
    Make sure you are in the directory where your main Streamlit script (e.g., `app.py`, `main.py`) is located.

2.  **Run the Streamlit application:**
    ```bash
    streamlit run poe_dashboard.py
    ```

    After running the command, your web browser should automatically open a new tab displaying your Streamlit application. If it doesn't, you can typically access it at `http://localhost:8501`.

---

## Project Structure

This project is organized as follows:

```
.
├── poe_dashboard.py      # The main Streamlit application
├── requirements.txt      # Lists all project dependencies
├── data/                 # Contains league data files
│   └── example_data.csv
└── tabs/                 # Holds separate Python files for each Streamlit tab/chart
    ├── tab_one_chart.py
    ├── tab_two_dashboard.py
    └── ...
```