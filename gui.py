import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os

class FileOpenerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Two File Opener")
        self.root.geometry("600x400")
        
        # Variables to store file paths
        self.file1_path = tk.StringVar()
        self.file2_path = tk.StringVar()
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # File 1 section
        ttk.Label(main_frame, text="File 1:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        file1_frame = ttk.Frame(main_frame)
        file1_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        file1_frame.columnconfigure(0, weight=1)
        
        self.file1_entry = ttk.Entry(file1_frame, textvariable=self.file1_path, state="readonly")
        self.file1_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(file1_frame, text="Browse...", 
                  command=self.browse_file1).grid(row=0, column=1)
        
        # File 2 section
        ttk.Label(main_frame, text="File 2:").grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        
        file2_frame = ttk.Frame(main_frame)
        file2_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        file2_frame.columnconfigure(0, weight=1)
        
        self.file2_entry = ttk.Entry(file2_frame, textvariable=self.file2_path, state="readonly")
        self.file2_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(file2_frame, text="Browse...", 
                  command=self.browse_file2).grid(row=0, column=1)
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=(0, 20))
        
        ttk.Button(button_frame, text="Process Files", 
                  command=self.process_files).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Clear All", 
                  command=self.clear_all).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Exit", 
                  command=self.root.quit).pack(side=tk.LEFT)
        
        # Info display area
        ttk.Label(main_frame, text="File Information:").grid(row=5, column=0, sticky=tk.W, pady=(0, 5))
        
        self.info_text = tk.Text(main_frame, height=10, width=70, wrap=tk.WORD)
        self.info_text.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Scrollbar for text area
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.info_text.yview)
        scrollbar.grid(row=6, column=2, sticky=(tk.N, tk.S), pady=(0, 10))
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        # Configure grid weights for the text area to expand
        main_frame.rowconfigure(6, weight=1)
    
    def browse_file1(self):
        """Open file dialog for first file"""
        filename = filedialog.askopenfilename(
            title="Select First File",
            filetypes=[
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.file1_path.set(filename)
            self.update_file_info()
    
    def browse_file2(self):
        """Open file dialog for second file"""
        filename = filedialog.askopenfilename(
            title="Select Second File",
            filetypes=[
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.file2_path.set(filename)
            self.update_file_info()
    
    def update_file_info(self):
        """Update the information display with file details"""
        info = []
        
        for i, file_path in enumerate([self.file1_path.get(), self.file2_path.get()], 1):
            if file_path:
                try:
                    file_size = os.path.getsize(file_path)
                    file_name = os.path.basename(file_path)
                    file_dir = os.path.dirname(file_path)
                    
                    info.append(f"File {i}:")
                    info.append(f"  Name: {file_name}")
                    info.append(f"  Directory: {file_dir}")
                    info.append(f"  Size: {file_size:,} bytes")
                    info.append("")
                except OSError as e:
                    info.append(f"File {i}: Error reading file - {e}")
                    info.append("")
            else:
                info.append(f"File {i}: No file selected")
                info.append("")
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, "\n".join(info))
    
    def process_files(self):
        """Process the selected files - customize this method for your needs"""
        file1 = self.file1_path.get()
        file2 = self.file2_path.get()
        
        if not file1 or not file2:
            messagebox.showwarning("Warning", "Please select both files before processing.")
            return
        
        try:
            # Example processing - just display first few lines of each file
            result = []
            result.append("=== FILE PROCESSING RESULTS ===\n")
            
            for i, file_path in enumerate([file1, file2], 1):
                result.append(f"File {i}: {os.path.basename(file_path)}")
                result.append("-" * 40)
                
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()[:5]  # Read first 5 lines
                    if lines:
                        result.extend(lines)
                        if len(f.readlines()) > 5:
                            result.append("... (file continues)")
                    else:
                        result.append("(empty file)")
                result.append("")
            
            # Display results in the info area
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, "".join(result))
            
            messagebox.showinfo("Success", "Files processed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error processing files: {str(e)}")
    
    def clear_all(self):
        """Clear all selections and info"""
        self.file1_path.set("")
        self.file2_path.set("")
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, "All selections cleared.")

def main():
    root = tk.Tk()
    app = FileOpenerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()