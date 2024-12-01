
## Setting Up the Environment from `environment.yml`

To ensure all dependencies are installed and configured correctly, use the `environment.yml` file to set up the Conda environment for this project.

---

### Steps to Load the Environment

1. **Ensure Conda is Installed**  
   Make sure you have [Conda](https://docs.conda.io/) installed. You can install either Miniconda or Anaconda.

2. **Navigate to the Project Directory**  
   Open a terminal and navigate to the folder containing the `environment.yml` file:
   ```bash
   cd /path/to/your/project
   ```

3. **Create the Conda Environment**
    Run the following command to create the environment:
    ```bash
   conda env create -f environment.yml
   ```
3. **Activate the Environment**
    Once the environment is created, activate it:
    ```bash
   conda activate hacker-cup
   ```

# Run the Script

use run.py in main_pipline

## Key Arguments

The script allows customization through the `Args` dataclass, enabling you to tailor the execution to specific needs. Below are the key configurable options:

| **Argument**                 | **Description**                                                                                 | **Default Value**      |
|------------------------------|-------------------------------------------------------------------------------------------------|------------------------|
| `problem_names`              | Names of the problems to solve.                                                                | Practice problems      |
| `problem_round`              | The round of problems to solve (e.g., `practice`, `round1`).                                   | `practice`             |
| `folder_path`                | Path to the folder containing problem files.                                                   | `./2024-practice`      |
| `timeout`                    | Timeout in seconds for each solution attempt.                                                  | `30`                   |
| `max_attempts`               | Maximum retries for refining and improving a solution.                                         | `20`                   |
| `few_shot_cot_examples_flag` | Include crafted few-shot Chain of Thought (CoT) examples in the prompts.                       | `True`                 |
| `retrive_flag`               | Use retrieval-augmented generation for additional problem context.                             | `False`                |
| `choose_best_flag`           | Enable multi-solution generation and ranking to select the best solution.                      | `False`                |
| `weave_log`                  | Enable logging in Weave for visualization and analysis.                                        | `True`                 |

## Acknowledgement
Our implementation is based on the [starter kit](https://github.com/HackerCupAI/starter-kits) offered by HackerCup AI 2024.
