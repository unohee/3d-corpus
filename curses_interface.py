#!/usr/bin/env python3
import os
import curses
import pickle
import sys
import importlib.util
import asyncio

def import_feature_extractor():
    """
    Dynamically import the featureExtractor_torch module.
    
    Returns:
        module: The imported featureExtractor_torch module
    """
    try:
        # Try direct import
        import featureExtractor_torch
        return featureExtractor_torch
    except ImportError:
        # Try dynamic import from current directory
        try:
            spec = importlib.util.spec_from_file_location("featureExtractor_torch", "./featureExtractor_torch.py")
            featureExtractor_torch = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(featureExtractor_torch)
            return featureExtractor_torch
        except Exception as e:
            print(f"Cannot import featureExtractor_torch module: {e}")
            sys.exit(1)

# Import module only when needed
featureExtractor_torch = None

class DatasetSelector:
    """
    TUI (Text User Interface) for selecting and processing dataset folders.
    
    This class provides a curses-based interface for browsing and selecting
    dataset folders to process with the feature extractor.
    """
    def __init__(self, screen, dataset_path="./dataset"):
        """
        Initialize the DatasetSelector.
        
        Args:
            screen: Curses screen object
            dataset_path: Path to the dataset directory
        """
        self.screen = screen
        self.dataset_path = dataset_path
        self.current_pos = 0
        self.offset = 0
        self.dirs = []
        self.all_items = []
        self.max_display = curses.LINES - 8
        
        # Screen setup
        curses.curs_set(0)  # Hide cursor
        curses.start_color()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Normal text
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Selected item
        curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Directory
        curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # PKL file
        curses.init_pair(5, curses.COLOR_RED, curses.COLOR_BLACK)    # Error message
        
        self.load_items()
        
    def load_items(self):
        """Load directory items from the dataset path."""
        self.dirs = []
        self.all_items = []
        
        try:
            items = os.listdir(self.dataset_path)
            
            # Show only directories
            for item in items:
                item_path = os.path.join(self.dataset_path, item)
                if os.path.isdir(item_path):
                    self.dirs.append(item)
            
            # Sort directories
            self.dirs.sort()
            
            # Combine all items (directories only)
            self.all_items = self.dirs
            
        except Exception as e:
            self.show_error(f"Error loading directory: {str(e)}")
    
    def draw(self):
        """Draw the interface on the screen."""
        self.screen.clear()
        h, w = self.screen.getmaxyx()
        
        # Display title
        title = "[ Dataset Folder Selector ]"
        self.screen.addstr(0, (w - len(title)) // 2, title, curses.color_pair(1) | curses.A_BOLD)
        
        # Display path
        path_text = f"Path: {self.dataset_path}"
        self.screen.addstr(2, 2, path_text, curses.color_pair(1))
        
        # Display help
        self.screen.addstr(3, 2, "Arrow keys: Move, Enter: Select/Enter, Space: Process folder, 'q': Quit, 'r': Refresh", curses.color_pair(1))
        
        # If no items
        if not self.all_items:
            self.screen.addstr(5, 2, "No folders found.", curses.color_pair(5))
            self.screen.refresh()
            return
        
        # Adjust display range
        if self.current_pos >= self.offset + self.max_display:
            self.offset = self.current_pos - self.max_display + 1
        elif self.current_pos < self.offset:
            self.offset = self.current_pos
        
        # Display items
        for i in range(min(self.max_display, len(self.all_items))):
            if self.offset + i >= len(self.all_items):
                break
                
            item = self.all_items[self.offset + i]
            
            # Determine item color
            if self.offset + i == self.current_pos:
                color = curses.color_pair(2)
            else:
                color = curses.color_pair(3)  # All directories
            
            # Add '/' to directory names
            display_name = item + '/'
            
            self.screen.addstr(5 + i, 2, display_name, color)
        
        # Display scrollbar (if items don't fit on screen)
        if len(self.all_items) > self.max_display:
            scroll_height = int(self.max_display * self.max_display / len(self.all_items))
            scroll_pos = int(self.offset * self.max_display / len(self.all_items))
            for i in range(self.max_display):
                if i >= scroll_pos and i < scroll_pos + scroll_height:
                    self.screen.addstr(5 + i, w - 2, "▓", curses.color_pair(1))
                else:
                    self.screen.addstr(5 + i, w - 2, "░", curses.color_pair(1))
        
        self.screen.refresh()
    
    def show_error(self, message):
        """
        Display an error message.
        
        Args:
            message: Error message to display
        """
        h, w = self.screen.getmaxyx()
        self.screen.addstr(h-2, 2, message, curses.color_pair(5))
        self.screen.refresh()
        curses.napms(2000)  # Show message for 2 seconds
    
    def show_info(self, message):
        """
        Display an info message.
        
        Args:
            message: Info message to display
        """
        h, w = self.screen.getmaxyx()
        self.screen.addstr(h-2, 2, " " * (w-4), curses.color_pair(1))  # Clear previous message
        self.screen.addstr(h-2, 2, message, curses.color_pair(4))
        self.screen.refresh()
    
    def process_folder(self, folder_path):
        """
        Process the selected folder by automatically loading audio files and extracting features.
        
        Args:
            folder_path: Path to the folder to process
            
        Returns:
            bool: True if refresh is needed, False otherwise
        """
        global featureExtractor_torch
        
        # Extract folder name
        folder_name = os.path.basename(folder_path)
        
        h, w = self.screen.getmaxyx()
        self.screen.clear()
        title = f"[ Processing folder {folder_name} ]"
        self.screen.addstr(0, (w - len(title)) // 2, title, curses.color_pair(1) | curses.A_BOLD)
        
        # Import module
        if featureExtractor_torch is None:
            self.screen.addstr(3, 2, "Loading module...", curses.color_pair(4))
            self.screen.refresh()
            featureExtractor_torch = import_feature_extractor()
        
        # Start automatic processing
        self.screen.clear()
        self.screen.addstr(0, (w - len(title)) // 2, title, curses.color_pair(1) | curses.A_BOLD)
        self.screen.addstr(3, 2, "1. Loading audio files...", curses.color_pair(4) | curses.A_BOLD)
        self.screen.refresh()
        
        # End curses interface before loading
        curses.endwin()
        
        try:
            print(f"=== Starting {folder_name} folder processing ===")
            print("1. Loading audio files...")
            
            # Load audio files
            buffers = asyncio.run(featureExtractor_torch.readfile_async(
                directory_path=folder_path,
                filename=folder_name,
                detect_onset=False,
                save_splitted_files=False
            ))
            
            print("2. Extracting features...")
            
            # Extract features
            asyncio.run(featureExtractor_torch.async_featureExtract(folder_name))
            
            print("3. Analyzing features...")
            
            # Analyze features
            feature_file = f"dataset/{folder_name}_features.pkl"
            featureExtractor_torch.check_features(feature_file)
            
            print("\nProcessing complete. Press any key to return to the interface...")
            input()
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print("Press any key to return to the interface...")
            input()
        
        # Restart interface
        return True  # Mark refresh needed
    
    def run(self):
        """Run the dataset selector interface."""
        refresh_needed = False
        
        while True:
            if refresh_needed:
                self.load_items()
                refresh_needed = False
                
            self.draw()
            
            # Get user input
            key = self.screen.getch()
            
            # Handle key
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.load_items()
            elif key == curses.KEY_UP:
                self.current_pos = max(0, self.current_pos - 1)
            elif key == curses.KEY_DOWN:
                self.current_pos = min(len(self.all_items) - 1, self.current_pos + 1)
            elif key == curses.KEY_PPAGE:  # Page Up
                self.current_pos = max(0, self.current_pos - self.max_display)
            elif key == curses.KEY_NPAGE:  # Page Down
                self.current_pos = min(len(self.all_items) - 1, self.current_pos + self.max_display)
            elif key == curses.KEY_HOME:
                self.current_pos = 0
            elif key == curses.KEY_END:
                self.current_pos = len(self.all_items) - 1
            elif key == ord(' ') or key == curses.KEY_ENTER or key == 10 or key == 13:
                if 0 <= self.current_pos < len(self.all_items):
                    selected_item = self.all_items[self.current_pos]
                    item_path = os.path.join(self.dataset_path, selected_item)
                    
                    if os.path.isdir(item_path):
                        # Process folder
                        self.show_info(f"Processing folder: {selected_item}")
                        refresh_needed = self.process_folder(item_path)
                    else:
                        self.show_error("Selected item is not a directory.")

def main(stdscr):
    """
    Main function for curses application.
    
    Args:
        stdscr: Curses screen object
    """
    selector = DatasetSelector(stdscr)
    selector.run()

def select_dataset():
    """
    Function to call externally: Run the dataset selection interface.
    
    Returns:
        Result from the curses wrapper
    """
    return curses.wrapper(main)

if __name__ == "__main__":
    curses.wrapper(main) 