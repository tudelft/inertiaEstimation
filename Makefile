
# change this to your likings
IN_PATH=input
OUT_PATH=output
CALIBRATION=final_calibration
SERIES=final_tests
CONFIGS=config_empty config_a config_b config_c

# do not change the following
PY=/usr/bin/env python3
SERIES_PATH=$(IN_PATH)/final_tests
CALIBRATION_PATH=$(IN_PATH)/final_calibration
CALIBRATION_OUT=$(OUT_PATH)/$(CALIBRATION).py
CALIBRATE=calibrate.py
ANALYSE=analyse_individually.py
TABLE=make_table.py

calibrate :
	@echo
	@echo "run calibration:"
	@echo
	$(PY) $(CALIBRATE) -vv --output $(CALIBRATION_OUT) $(CALIBRATION_PATH)

analyse : calibrate
	@echo
	@echo "processing configurations:"
	@echo $(CONFIGS)
	@echo
	$(foreach var,$(CONFIGS), \
		$(PY) $(ANALYSE) -vv --plots --output $(OUT_PATH)/$(SERIES) $(SERIES_PATH)/$(var) $(CALIBRATION_OUT); \
	)

table : analyse
	@echo
	@echo "making table for configurations:"
	@echo $(CONFIGS)
	@echo
	$(PY) $(TABLE) $(foreach var,$(CONFIGS), $(OUT_PATH)/$(SERIES)/$(var).pkl)
#	ipython -i $(TABLE) -- $(foreach var,$(CONFIGS), $(OUT_PATH)/$(SERIES)/$(var).pkl)

all : table
