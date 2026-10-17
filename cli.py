"""
Module: cli.py
Deskripsi: Alias antarmuka baris perintah (CLI) yang meneruskan eksekusi secara terintegrasi ke main.py.
           Memenuhi seluruh fitur analisis foto tunggal, video tracking, moderasi, dan ekspor.
"""

from main import main, analyze_single_photo, display_interactive_menu, parse_arguments

if __name__ == "__main__":
    main()
