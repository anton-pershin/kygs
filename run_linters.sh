black kygs/

isort kygs/

printf "\nPress any key to continue to pylint...\n"
read -n 1 -s -r
pylint kygs/

printf "\nPress any key to continue to mypy...\n"
read -n 1 -s -r
mypy kygs/
