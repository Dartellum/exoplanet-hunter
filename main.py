# main.py
import os
import sys
import tess_survey
import wasp_survey
import radial_velocity_survey

def clear_screen():
    """Clears the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def main_menu():
    """Displays the main menu and handles user input."""
    while True:
        clear_screen()
        print("\n==============================")
        print(" 🔭 EXOPLANET HUNTER SUITE 🔭")
        print("==============================")
        print("\nHigh-Level Menu:")
        print("  1. TESS Survey")
        print("  2. WASP Survey")
        print("  3. Radial Velocity Survey")
        print("  4. Exit")

        choice = input("\nSelect an option: ")

        if choice == '1':
            tess_survey.menu()
        elif choice == '2':
            wasp_survey.menu()
        elif choice == '3':
            radial_velocity_survey.menu()
        elif choice == '4':
            print("\nExiting. Happy hunting! 🔭")
            sys.exit()
        else:
            print("\nInvalid choice. Please try again.")
            input("Press Enter to continue...")

if __name__ == '__main__':
    main_menu()
