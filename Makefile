##
## EPITECH PROJECT, 2026
## MY_TORCH
## File description:
## Unified Makefile for my_torch project
##

ANALYZER	=	my_torch_analyzer
GENERATOR	=	my_torch_generator
VISU		=	visu

SRC_ANALYZER	=	src/cli/analyzer.py
SRC_GENERATOR	=	src/cli/generator.py
SRC_VISU	=	bonus/visualize.py

all: $(ANALYZER) $(GENERATOR)

$(ANALYZER): $(SRC_ANALYZER) Makefile
	@printf '%s\n' '#!/usr/bin/env python3' 'import os, sys' 'sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))' 'from src.cli.analyzer import main' 'if __name__ == "__main__":' '    main()' > $(ANALYZER)
	@chmod +x $(ANALYZER)
	@echo "[OK] $(ANALYZER) ready."

$(GENERATOR): $(SRC_GENERATOR) Makefile
	@printf '%s\n' '#!/usr/bin/env python3' 'import os, sys' 'sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))' 'from src.cli.generator import main' 'if __name__ == "__main__":' '    main()' > $(GENERATOR)
	@chmod +x $(GENERATOR)
	@echo "[OK] $(GENERATOR) ready."

visualize: $(VISU)

$(VISU): $(SRC_VISU) Makefile
	@printf '%s\n' '#!/usr/bin/env python3' 'import os, sys' 'sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))' 'from bonus.visualize import main' 'if __name__ == "__main__":' '    sys.exit(main())' > $(VISU)
	@chmod +x $(VISU)
	@echo "[OK] $(VISU) ready."

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@rm -rf .pytest_cache .coverage htmlcov 2>/dev/null || true
	@echo "[OK] Cleaned temporary files."

fclean: clean
	@rm -f $(ANALYZER) $(GENERATOR) $(VISU) bonus/play bonus/visu
	@echo "[OK] Cleaned deliverable executables."

re: fclean all

tests_run: all
	@python3 -m unittest discover -s tests -p "test_*.py"

.PHONY: all clean fclean re tests_run visualize
