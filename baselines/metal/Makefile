BUILD := build
RUNNER := $(BUILD)/run

.PHONY: all check selftest clean

all: $(RUNNER)

$(RUNNER): run.m
	@mkdir -p $(BUILD)
	clang -O2 -Wall -Wextra -fobjc-arc -framework Foundation -framework Metal $< -o $@

selftest: $(RUNNER)
	$(RUNNER) --selftest

check: $(RUNNER)
	$(RUNNER) --kernels kernels

clean:
	rm -rf $(BUILD)
