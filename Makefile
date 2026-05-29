.PHONY: help test run flatpak flatpak-install flatpak-bundle flatpak-run flatpak-clean

APP_ID := io.github.jeroenvdwaal.StickerCreator
BUILD  := packaging/flatpak/build.sh

help:
	@echo "Targets:"
	@echo "  test            Run pytest suite"
	@echo "  run             Run the app from source (python3 main.py)"
	@echo "  flatpak         Build the Flatpak (no install)"
	@echo "  flatpak-install Build + install into user remote"
	@echo "  flatpak-bundle  Build + export single-file .flatpak"
	@echo "  flatpak-run     Run installed Flatpak"
	@echo "  flatpak-clean   Remove build-dir, state-dir, repo"

test:
	pytest

run:
	python3 main.py

flatpak:
	$(BUILD)

flatpak-install:
	INSTALL=1 $(BUILD)

flatpak-bundle:
	BUNDLE=1 $(BUILD)

flatpak-run:
	flatpak run $(APP_ID)

flatpak-clean:
	rm -rf packaging/flatpak/build-dir \
	       packaging/flatpak/.flatpak-builder \
	       packaging/flatpak/repo \
	       packaging/flatpak/$(APP_ID).flatpak
