##
## EPITECH PROJECT, 2026
## MY_TORCH_ - BONUS
## File description:
## Dedicated Makefile for all bonus features
##

PLAY_BIN	=	play
VISU_BIN	=	visu

all: $(PLAY_BIN) $(VISU_BIN)

$(PLAY_BIN): play_ai.py Makefile
	@printf '%s\n' '#!/usr/bin/env python3' 'import os, sys' 'sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))' 'from bonus.play_ai import main' 'if __name__ == "__main__":' '    main()' > $(PLAY_BIN)
	@chmod +x $(PLAY_BIN)
	@echo "[OK] bonus/$(PLAY_BIN) ready."

$(VISU_BIN): visualize.py Makefile
	@printf '%s\n' '#!/usr/bin/env python3' 'import os, sys' 'sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))' 'from bonus.visualize import main' 'if __name__ == "__main__":' '    sys.exit(main())' > $(VISU_BIN)
	@chmod +x $(VISU_BIN)
	@echo "[OK] bonus/$(VISU_BIN) ready."

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "[OK] Cleaned bonus temporary files."

fclean: clean
	@rm -f $(PLAY_BIN) $(VISU_BIN)
	@echo "[OK] Cleaned bonus executables."

re: fclean all

test: all
	@python3 play_ai.py --auto 2
	@echo "[OK] Bonus test passed successfully."

.PHONY: all clean fclean re test
