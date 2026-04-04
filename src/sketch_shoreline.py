import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Frame, Label, Button, Checkbutton, BooleanVar, StringVar, OptionMenu, IntVar

import cv2
import numpy as np
import os
import json
import re
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon
# Add to imports at the top
from matplotlib.figure import Figure
from shapely.geometry import Polygon as ShapelyPolygon, LineString

def convert_str_to_datetime(filenames: list, reverse: bool = False):
	if not reverse:
		ls = filenames
		dt_components = [re.search(r'(\d{4}-\d{2}-\d{2}_\d{4})', f).group(1) for f in ls]
		dt_objs = [datetime.strptime(dt, "%Y-%m-%d_%H%M") for dt in dt_components]
		print(f"Datetime Objects:\n{dt_objs[:5]}")
		return dt_objs
	else:
		ls = filenames
		dt_components = [dt.strftime("%Y-%m-%d_%H%M") for dt in ls]
		print(f"Datetime Filename String Component:\n{dt_components[:5]}")
		return dt_components
	

class ShorelineSketch:
	def __init__(self, root, station):
		self.root = root
		self.root.title("Shoreline Annotator")
		self.root.state('zoomed') # maximize window
		self.station = station
		self.setup_variables()
		self.setup_menu()
		self.setup_ui()
		self.load_image_list()

	def setup_variables(self):
		# Initialize Variables
		cwd = Path(os.getcwd())
		print(f"CWD: {cwd}")

		# Station Configs Path Directory
		station_config_path = cwd / "src" / rf"{self.station}.config.json"
		print(f"Station Config Path: {station_config_path}")

		# Stored Time Average Images Directories
		images_dir = cwd / "images" 
		self.timeavg_images_dir = images_dir / rf"{self.station}" / "time_average" / "n=100-frames"
		self.time_avg_images = os.listdir(self.timeavg_images_dir)
		self.dt_objects = convert_str_to_datetime(self.time_avg_images)
		self.dt_strings = convert_str_to_datetime(self.dt_objects, reverse=True)

		# Stored Annotations
		self.annotations_file = cwd / "data" / "annotations" / rf"{self.station}_annotations.jsonl"

		


	def setup_menu(self):
		# Create Menu
		menubar = tk.Menu(self.root)
		filemenu = tk.Menu(menubar, tearoff=0)
		filemenu.add_command(label="Exit", command=self.root.quit)
		menubar.add_cascade(label="File", menu=filemenu)
		self.root.config(menu=menubar)


	def setup_ui(self):
		# Create control panel
		self.control_frame = tk.Frame(self.root)
		self.control_frame.pack(fill=tk.X, padx=5, pady=5)
	
		self.curve_btn = tk.Button(self.control_frame, text="Draw Curve", 
								command=self.start_curve_drawing)
		self.curve_btn.pack(side=tk.LEFT, padx=5)

		# Add state variables
		self.drawing_curve = False
		self.curve_points = []  # List of (x,y) tuples
	
		self.save_btn = tk.Button(self.control_frame, text="Save Annotations", 
								command=self.save_annotations)
		self.save_btn.pack(side=tk.LEFT, padx=5)

		# reset view
		self.reset_view_btn = tk.Button(self.control_frame, text="Reset View", 
									command=self.reset_view)
		self.reset_view_btn.pack(side=tk.LEFT, padx=5)

		# Create main content frame to hold left panel and image display
		self.main_content = tk.Frame(self.root)
		self.main_content.pack(fill=tk.BOTH, expand=True)

		# Left panel for image list
		self.left_panel = tk.Frame(self.main_content, width=400, bg='lightgray')
		self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0), pady=5)
		self.left_panel.pack_propagate(False)  # Fixed width

		# Label for the list
		tk.Label(self.left_panel, text="Time-Average Images:", bg='lightgray').pack(pady=(5, 0))

		# Create frame for image display
		self.image_frame = tk.Frame(self.main_content)
		self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
		self.fig, self.ax = plt.subplots(figsize=(10, 10))
		self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
		self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
		self.canvas.mpl_connect('scroll_event', self.on_scroll)
		self.canvas.mpl_connect('button_press_event', self.on_click)


		# Create frame for listbox and scrollbar
		list_frame = tk.Frame(self.left_panel)
		list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

		# Scrollbar
		scrollbar = tk.Scrollbar(list_frame)
		scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

		# Listbox
		# Configure a custom font for the listbox
		listbox_font = ('Segoe UI', 12)  # or ('Arial', 11)
		self.image_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, 
										selectmode=tk.SINGLE, bg='white',
										font=listbox_font)
		self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

		# Configure scrollbar
		scrollbar.config(command=self.image_listbox.yview)

	def load_image_list(self):
		# Load all existing annotations
		existing_annotations = set()
		if self.annotations_file.exists():
			with open(self.annotations_file, 'r') as f:
				for line in f:
					try:
						annotation = json.loads(line.strip())
						existing_annotations.add(annotation.get("timestamp"))
					except json.JSONDecodeError:
						continue
		
		# Populate listbox with images
		for idx, img in enumerate(self.time_avg_images):
			dt_str = self.dt_strings[idx]
			if dt_str in existing_annotations:
				display_text = f"✅  {self.station}-{dt_str}"
			else:
				display_text = f"⭕  {self.station}-{dt_str}"
			self.image_listbox.insert(tk.END, display_text)

		# Bind selection event and display image
		self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)

		# Select first item
		self.image_listbox.selection_set(0)
		self.image_listbox.event_generate("<<ListboxSelect>>")
	
	def refresh_image_list(self):
		"""Reload the listbox to update checkmark indicators"""
		# Store current selection
		current_selection = self.image_listbox.curselection()
		
		# Clear and repopulate
		self.image_listbox.delete(0, tk.END)
		
		# Load all existing annotations
		existing_annotations = set()
		if self.annotations_file.exists():
			with open(self.annotations_file, 'r') as f:
				for line in f:
					try:
						annotation = json.loads(line.strip())
						existing_annotations.add(annotation.get("timestamp"))
					except json.JSONDecodeError:
						continue
		
		# Repopulate
		for idx, img in enumerate(self.time_avg_images):
			dt_str = self.dt_strings[idx]
			if dt_str in existing_annotations:
				display_text = f"✅  {self.station}-{dt_str}"
			else:
				display_text = f"⭕  {self.station}-{dt_str}"
			self.image_listbox.insert(tk.END, display_text)
		
		# Restore selection if possible
		if current_selection:
			self.image_listbox.selection_set(current_selection[0])

	# display image on select
	def on_image_select(self, event):
		selection = self.image_listbox.curselection()
		if selection:
			index = selection[0]
			selected_image = self.time_avg_images[index]
			print(f"Selected: {selected_image}")
			# Load and display the selected image
			self.load_image(os.path.join(self.timeavg_images_dir, selected_image))

	def load_image(self, filepath):
		"""Load and display the selected image"""
		# Clear curve points when loading new image
		self.curve_points = []
		if hasattr(self, 'drawing_curve') and self.drawing_curve:
			self.drawing_curve = False
			self.curve_btn.config(relief=tk.RAISED)
		# Reset view flag for new image
		self._view_initialized = False

		try:
			# Read image with OpenCV
			img = cv2.imread(filepath)
			if img is None:
				messagebox.showerror("Error", f"Could not open image at {filepath}")
				return
			
			# Convert from BGR to RGB for matplotlib
			self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			self.current_image_path = filepath
			
			# Update display
			self.update_display()
			
			# Update window title
			self.root.title(f"Shoreline Sketch - {os.path.basename(filepath)}")
			
		except Exception as e:
			messagebox.showerror("Error", f"Failed to load image: {str(e)}")
	
	def update_display(self):
		# Store current view limits only if we're not on first load
		if hasattr(self, '_view_initialized') and self._view_initialized:
			xlim = self.ax.get_xlim()
			ylim = self.ax.get_ylim()
			restore_limits = True
		else:
			restore_limits = False
		

		print(f"Debug: _view_initialized={hasattr(self, '_view_initialized') and self._view_initialized}, restore_limits={restore_limits}")

		self.ax.clear()
		
		if self.image is not None:
			self.ax.imshow(self.image)
			
			if not self._view_initialized:
				# First load - let matplotlib auto-set limits
				self._view_initialized = True
			elif restore_limits:
				# Subsequent updates - restore previous view
				self.ax.set_xlim(xlim)
				self.ax.set_ylim(ylim)
		
		# Draw curve points
		if self.curve_points:
			points = np.array(self.curve_points)
			self.ax.plot(points[:,0], points[:,1], 'o-', color='cyan', linewidth=2, markersize=4)
		
		self.ax.axis('off')
		self.canvas.draw()
	
	
	def toggle_pan_mode(self):
		self.pan_mode = not self.pan_mode
		self.pan_button.config(text=f"Pan Mode: {'ON' if self.pan_mode else 'OFF'}")
	
	def on_scroll(self, event):
		"""Zoom in/out with mouse wheel"""
		if event.inaxes != self.ax:
			return
		
		# Get current limits
		xlim = self.ax.get_xlim()
		ylim = self.ax.get_ylim()
		
		# Calculate zoom factor
		scale_factor = 1.1 if event.button == 'down' else 0.9
		
		# Get mouse position
		x = event.xdata
		y = event.ydata
		
		# Set new limits
		self.ax.set_xlim([x - (x - xlim[0]) * scale_factor, 
						x + (xlim[1] - x) * scale_factor])
		self.ax.set_ylim([y - (y - ylim[0]) * scale_factor, 
						y + (ylim[1] - y) * scale_factor])
		
		self.canvas.draw()
	
	
	def reset_view(self):
		"""Reset zoom to show full image"""
		if self.image is not None:
			self.ax.set_xlim(0, self.image.shape[1])
			self.ax.set_ylim(self.image.shape[0], 0)  # Reverse for image coordinates
			self.canvas.draw()
	
	def start_curve_drawing(self):
		self.drawing_curve = True
		self.curve_points = []
		self.curve_btn.config(relief=tk.SUNKEN)
		self.update_display()

	def on_click(self, event):
		if not self.drawing_curve or event.inaxes != self.ax:
			return
		
		if event.button == 1:  # Left click - add point
			self.curve_points.append((event.xdata, event.ydata))
			self.update_display()
			
		elif event.button == 3:  # Right click - remove last point
			if self.curve_points:
				self.curve_points.pop()
				self.update_display()

	def finish_curve(self):
		"""Call this when saving or switching modes"""
		self.drawing_curve = False
		self.curve_btn.config(relief=tk.RAISED)
	
	def save_annotations(self):
		"""Save curve points to station annotations file"""
		if not self.curve_points:
			messagebox.showwarning("Warning", "No points to save")
			return
		
		# Get current image timestamp
		current_file = os.path.basename(self.current_image_path)
		match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{4})', current_file)
		if not match:
			messagebox.showerror("Error", "Could not extract timestamp from filename")
			return
		
		timestamp = match.group(1)
		
		# Path to annotations file
		annotations_file = Path(os.getcwd()) / "data" / "annotations" / f"{self.station}_annotations.jsonl"
		annotations_file.parent.mkdir(exist_ok=True)
		
		# Load all existing annotations
		all_annotations = []
		if annotations_file.exists():
			with open(annotations_file, 'r') as f:
				for line in f:
					try:
						ann = json.loads(line.strip())
						# Keep all except the current timestamp
						if ann.get("timestamp") != timestamp:
							all_annotations.append(ann)
					except json.JSONDecodeError:
						continue
		
		# Add new/updated annotation
		all_annotations.append({
			"timestamp": timestamp,
			"image_file": current_file,
			"points": [[round(x, 2), round(y, 2)] for x, y in self.curve_points]
		})
		
		# Write all back
		with open(annotations_file, 'w') as f:
			for ann in all_annotations:
				f.write(json.dumps(ann) + '\n')
		
		# Refresh listbox
		self.refresh_image_list()
		messagebox.showinfo("Success", f"Saved {len(self.curve_points)} points for {timestamp}")



if __name__ == "__main__":
	root = tk.Tk()
	station = "jennette_south"
	app = ShorelineSketch(root, station)
	root.mainloop()
	