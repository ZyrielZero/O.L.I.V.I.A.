"""Flet desktop app - main entry."""

import logging

import flet as ft

from src.flet_app.app import OliviaApp

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main(page: ft.Page):
    """Flet entry point."""
    OliviaApp(page)


if __name__ == "__main__":
    print("=" * 60)
    print("  O.L.I.V.I.A. Desktop")
    print("=" * 60)
    ft.run(main)
