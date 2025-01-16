
SERIES=cyberzoo_tests_the_second
CONFIGS=config_empty config_a config_b config_c

PY=/usr/bin/env python3
ANALYSE=analyse_individually.py
FIGURES=make_figures.py

analyse :
	@echo
	@echo "processing configurations:"
	@echo $(CONFIGS)
	@echo
	@$(foreach var,$(CONFIGS), \
		$(PY) $(ANALYSE) input/$(SERIES)/$(var) input/$(SERIES)/calibration input/$(SERIES)/device --output output/$(SERIES); \
	)

figures :
	@echo
	@echo "making figures for configurations:"
	@echo $(CONFIGS)
	@echo
#	@$(PY) $(FIGURES) $(foreach var,$(CONFIGS), output/$(SERIES)/$(var).pkl)
	@ipython -i $(FIGURES) -- $(foreach var,$(CONFIGS), output/$(SERIES)/$(var).pkl)

all : analyse figures
