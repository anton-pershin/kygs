# AGENTS.md

## Setup commands

- Set up the environment: `source set_env.sh`
- Run the executable script you are interested in: `python kygs/scripts/XXX.py`
- Run tests: `pytest`

## Executable script configuration

- Any executable script is configured via hydra in `config/config_XXX.yaml`
- In general, all the user-specific fields (API tokens, paths etc.) should be stored in `config/user_settings/user_settings.yaml`. Do not commit `user_settings.yaml` without explicit permission

## Developer guide

### Basics

- All the code structure is documented well in `README.md` and `docs/`. Read the documentations carefully before making any changes.
- This code has multiple entry points all of which are called "executable scripts" and are located in `kygs/scripts`. Depending on the task, one or another executable script has to be invoked.
- This code is configured using hydra. Its configs can be found in `config/`. No matter what constants/literals are used, they should be taken from configs. Object construction can also be made via `hydra.utils.instantiate` but use it only with simple classes (i.e., not derived from some base class).

### Workflow

- Before making any changes, create a local git branch named `YYYYMMDD_short_task_description` where `YYYYMMDD` stands for the current date. Checkout this local branch and make all the changes there. Do not forget to make occasional commits during your work to be able to roll back to previous versions of the code if necessary. NEVER commit `config/user_settings/user_settings.yaml`, keep its changes unstaged.
- Run the following linters before finishing the job:
  - black to ensure good formatting (note that it changes the code): `black kygs/`
  - isort to ensure good import sorting (note that it changes the code): `isort kygs/`
  - pylint to ensure typing: `pylint kygs/`
  - mypy to ensure typing: `mypy kygs/`
- Use type hints, their use is necessiated by linters
- Update `README.md` and `docs/` when new functionality is added or there is outdated information

### Fundamental classes

- `Message` - a fundamental class primarily storing the message text, date and author. In this repo, we assume that we always manipulate messages to solve a specific text analysis problem so all the texts should be initially obtained through this class' instances
- `MessageProvider` - a fundamental class loading and storing a collection of messages. E.g., a jsonl file storing reddit posts should eventually become a `MessageProvider` in the code
